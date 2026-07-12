# ============================================================
# Ablation experiments: หาว่าอะไรทำให้คะแนนต่างจาก Kaggle RF
# รันบน Kaggle notebook ได้เลย (ใช้ fraudTrain.csv / fraudTest.csv)
#
# สมมติฐานที่จะทดสอบ:
#   H1: target-encoded rates (state/city/job/merchant) เป็นพิษ (in-train leakage -> noise ใน test)
#   H2: Month ทำร้ายโมเดลเพราะ test = ก.ค.-ธ.ค. 2020 ล้วนๆ
#   H3: ฟีเจอร์ geo ดิบ (lat/long/zip/city_pop/merch coords) ช่วย (customer identity)
#   H4: RF โตเต็มต้น (default) เหมาะกับ synthetic data นี้
# ============================================================

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve

TRAIN_PATH = "fraudTrain.csv"
TEST_PATH = "fraudTest.csv"


# ---------- โหลด + ฟีเจอร์พื้นฐาน (เหมือนของคุณ แต่ไม่ drop อะไรทิ้ง) ----------
def load_and_prepare(path):
    df = pd.read_csv(path, index_col=0)
    dt = pd.to_datetime(df["trans_date_trans_time"])
    df["Hour"] = dt.dt.hour
    df["Day"] = dt.dt.dayofweek
    df["Month"] = dt.dt.month
    df["is_night"] = (~df["Hour"].between(5, 20)).astype(int)
    df["age"] = 2021 - pd.to_datetime(df["dob"]).dt.year
    df["is_online"] = df["category"].str.endswith("_net").astype(int)
    return df, dt


def add_velocity(df, dt):
    """Velocity features แบบเดียวกับของคุณ (ตรวจแล้วว่า alignment ถูกต้อง)"""
    df = df.copy()
    df["datetime"] = dt.values
    df = df.sort_values(["cc_num", "datetime"]).reset_index(drop=True)
    df["time_since_last"] = (
        df.groupby("cc_num")["datetime"].diff().dt.total_seconds().fillna(999999)
    )
    r1h = df.groupby("cc_num").rolling("1h", on="datetime")["amt"]
    df["txn_count_1h"] = r1h.count().values
    df["amt_sum_1h"] = r1h.sum().values
    r24 = df.groupby("cc_num").rolling("24h", on="datetime")["amt"]
    df["amt_mean_24h"] = r24.mean().values
    df["amt_vs_mean_24h"] = df["amt"] / df["amt_mean_24h"].replace(0, 1)
    return df.drop(columns=["datetime"])


# ---------- per-card profile จาก "ประวัติใน train เท่านั้น" (H3 แบบสะอาด) ----------
# แทนที่ lat/long-as-identity ด้วยสถิติรายใบเต็มช่วง train แล้ว merge เข้า test ด้วย cc_num
# (ลูกค้า 1000 คนเดิมอยู่ทั้งสองไฟล์ -> deploy จริงก็ทำแบบนี้ได้: ธนาคารรู้จักลูกค้าตัวเอง)
def add_card_profile(train, test):
    prof = (
        train.groupby("cc_num")["amt"]
        .agg(card_amt_mean="mean", card_amt_std="std", card_txn_cnt="count")
        .reset_index()
    )
    prof["card_amt_std"] = prof["card_amt_std"].fillna(prof["card_amt_std"].median())
    out = []
    for d in (train, test):
        d = d.merge(prof, on="cc_num", how="left")
        d["amt_z_vs_card"] = (d["amt"] - d["card_amt_mean"]) / (
            d["card_amt_std"] + 1e-6
        )
        out.append(d)
    return out


# ---------- target encoding แบบ fit-on-full-train (วิธีเดิมของคุณ) ----------
def target_encode_naive(train, test, cols, target="is_fraud", min_samples=50, k=100):
    train, test = train.copy(), test.copy()
    g = train[target].mean()
    for col in cols:
        s = train.groupby(col)[target].agg(["mean", "count"])
        enc = ((s["count"] * s["mean"] + k * g) / (s["count"] + k)).where(
            s["count"] >= min_samples, g
        )
        train[f"{col}_fr"] = train[col].map(enc).fillna(g)
        test[f"{col}_fr"] = test[col].map(enc).fillna(g)
    return train, test


# ---------- target encoding แบบ out-of-fold ด้วย GroupKFold ตาม cc_num ----------
# ปิดช่อง "episode ของเหยื่อคนเดียวกันรั่วข้าม fold"
from sklearn.model_selection import GroupKFold


def target_encode_oof(
    train, test, cols, target="is_fraud", min_samples=50, k=100, n_splits=5
):
    train, test = train.copy(), test.copy()
    g = train[target].mean()
    gkf = GroupKFold(n_splits=n_splits)
    for col in cols:
        new = f"{col}_fr"
        train[new] = g
        for fit_idx, val_idx in gkf.split(train, groups=train["cc_num"]):
            fit = train.iloc[fit_idx]
            s = fit.groupby(col)[target].agg(["mean", "count"])
            enc = ((s["count"] * s["mean"] + k * g) / (s["count"] + k)).where(
                s["count"] >= min_samples, g
            )
            train.iloc[val_idx, train.columns.get_loc(new)] = (
                train.iloc[val_idx][col].map(enc).fillna(g).values
            )
        s = train.groupby(col)[target].agg(["mean", "count"])
        enc = ((s["count"] * s["mean"] + k * g) / (s["count"] + k)).where(
            s["count"] >= min_samples, g
        )
        test[new] = test[col].map(enc).fillna(g)
    return train, test


# ---------- ประเมิน: PR-AUC + จุด F1 สูงสุดบน PR curve ----------
def evaluate(name, y_test, y_prob):
    ap = average_precision_score(y_test, y_prob)
    prec, rec, thr = precision_recall_curve(y_test, y_prob)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    i = np.nanargmax(f1)
    print(
        f"{name:32s} | PR-AUC {ap:.4f} | best-F1 {f1[i]:.3f} "
        f"(P {prec[i]:.3f}, R {rec[i]:.3f}, thr~{thr[min(i, len(thr) - 1)]:.3f})"
    )
    return ap


def run_lgbm(name, Xtr, ytr, Xte, yte):
    m = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    m.fit(Xtr, ytr)
    return evaluate(name, yte, m.predict_proba(Xte)[:, 1]), m


# ============================================================
# เตรียมข้อมูล
# ============================================================
train, dt_tr = load_and_prepare(TRAIN_PATH)
test, dt_te = load_and_prepare(TEST_PATH)
train = add_velocity(train, dt_tr)
test = add_velocity(test, dt_te)
train, test = add_card_profile(train, test)

cat_tr = pd.get_dummies(train["category"], prefix="cat", dtype=int)
cat_te = pd.get_dummies(test["category"], prefix="cat", dtype=int).reindex(
    columns=cat_tr.columns, fill_value=0
)

BASE = ["amt", "Hour", "Day", "is_night", "age", "is_online"]  # แกนกลางที่ transfer แน่
VELO = [
    "time_since_last",
    "txn_count_1h",
    "amt_sum_1h",
    "amt_mean_24h",
    "amt_vs_mean_24h",
]
GEO = ["zip", "lat", "long", "city_pop", "merch_lat", "merch_long"]  # แบบ Kaggle
CARD = ["card_amt_mean", "card_amt_std", "card_txn_cnt", "amt_z_vs_card"]
ENC = ["state_fr", "city_fr", "job_fr", "merchant_fr"]

tr_naive, te_naive = target_encode_naive(
    train, test, ["state", "city", "job", "merchant"]
)
tr_oof, te_oof = target_encode_oof(train, test, ["state", "city", "job", "merchant"])

y_tr, y_te = train["is_fraud"], test["is_fraud"]


def X(df_num, cats, cols):
    return pd.concat(
        [df_num[cols].reset_index(drop=True), cats.reset_index(drop=True)], axis=1
    )


print("=" * 90)
# A) จำลองชุดฟีเจอร์ Kaggle (ดิบล้วน + Month) ด้วย LightGBM ของคุณ
run_lgbm(
    "A. Kaggle-style raw features",
    X(train, cat_tr, ["amt"] + GEO + ["age", "Hour", "Day", "Month"]),
    y_tr,
    X(test, cat_te, ["amt"] + GEO + ["age", "Hour", "Day", "Month"]),
    y_te,
)

# B) ประมาณชุดปัจจุบันของคุณ (มี encoding แบบเดิม + Month, ไม่มี geo)
run_lgbm(
    "B. Your current (naive enc+Month)",
    X(tr_naive, cat_tr, BASE + VELO + ["Month"] + ENC),
    y_tr,
    X(te_naive, cat_te, BASE + VELO + ["Month"] + ENC),
    y_te,
)

# C) ของคุณ "ถอนพิษ": ตัด encoding ทั้ง 4 + ตัด Month   <-- ทดสอบ H1+H2
run_lgbm(
    "C. Yours minus enc minus Month",
    X(train, cat_tr, BASE + VELO),
    y_tr,
    X(test, cat_te, BASE + VELO),
    y_te,
)

# D) C + OOF encoding (ถ้าอยากเก็บ encoding ไว้)      <-- ทดสอบ H1 เชิงวิธีแก้
run_lgbm(
    "D. C + OOF(GroupKFold) encoding",
    X(tr_oof, cat_tr, BASE + VELO + ENC),
    y_tr,
    X(te_oof, cat_te, BASE + VELO + ENC),
    y_te,
)

# E) C + geo ดิบ + per-card profile                    <-- ทดสอบ H3
run_lgbm(
    "E. C + geo + card profile",
    X(train, cat_tr, BASE + VELO + GEO + CARD),
    y_tr,
    X(test, cat_te, BASE + VELO + GEO + CARD),
    y_te,
)

# F) จำลอง RF ของโน้ตบุ๊ก Kaggle (default = โตเต็มต้น)  <-- ทดสอบ H4
#    หมายเหตุ: เขาเทรนบน SMOTE; ที่นี่ใช้ class_weight แทนเพื่อประหยัด RAM/เวลา
rf = RandomForestClassifier(
    n_estimators=100, class_weight="balanced_subsample", random_state=5, n_jobs=-1
)
Xtr_rf = X(train, cat_tr, ["amt"] + GEO + ["age", "Hour", "Day", "Month"])
Xte_rf = X(test, cat_te, ["amt"] + GEO + ["age", "Hour", "Day", "Month"])
rf.fit(Xtr_rf, y_tr)
evaluate("F. Full-depth RF (Kaggle-style)", y_te, rf.predict_proba(Xte_rf)[:, 1])

# ============================================================
# วิธีอ่านผล:
#   C >> B มาก        -> ยืนยัน H1/H2: encoding เดิม + Month คือตัวถ่วง
#   D ~ C หรือดีกว่านิด -> OOF ปลอดภัยแล้ว แต่ค่าเพิ่มอาจน้อย (ทิ้งไปเลยก็ได้)
#   E > C              -> geo/card-profile มีค่า (H3) ควรใส่กลับ
#   A, F สูง           -> ฟีเจอร์ดิบ + ความจุโมเดล คือแหล่งพลังของโน้ตบุ๊ก Kaggle
# ต่อยอด: ทำ time-based validation ใน train (เช่น validate ด้วย มี.ค.-มิ.ย. 2020)
# ก่อนตัดสินใจ keep/drop ฟีเจอร์ใดๆ ในอนาคต
# ============================================================
