# 💳 Credit Card Fraud Detection

Binary classification model to detect fraudulent credit card transactions using behavioral, temporal, and target-encoded features. Evaluated on PR-AUC to handle severe class imbalance, with a three-scenario business case analysis across different deployment thresholds.

---

## 📌 Problem Statement

Credit card fraud datasets are highly imbalanced — fraudulent transactions represent a tiny fraction of all activity. Standard accuracy metrics fail in this setting. This project builds a detection pipeline that prioritizes catching fraud (high Recall) while minimizing false alerts (high Precision), using **PR-AUC** as the primary evaluation metric.

![Class Imbalance](images/plot_fraud_class_imbalance.png)

---

## 📂 Dataset

**Source:** [Fraud Detection Dataset — Kaggle (kartik2112)](https://www.kaggle.com/datasets/kartik2112/fraud-detection)

| Split | Rows | Period |
|---|---|---|
| Train | fraudTrain.csv | Jan 2019 – Jun 2020 |
| Test | fraudTest.csv | Jul 2020 – Dec 2020 |

---

## 🔧 Feature Engineering

### Velocity Features *(most impactful)*
Aggregated per credit card number over rolling time windows:

| Feature | Description |
|---|---|
| `time_since_last` | Seconds since the card's last transaction |
| `txn_count_1h` | Number of transactions in the past 1 hour |
| `amt_sum_1h` | Total amount spent in the past 1 hour |
| `amt_mean_24h` | Rolling 24-hour average amount |
| `amt_vs_mean_24h` | Ratio of current amount vs 24h average |

> **Why velocity?** Raw amount tells you "this transaction is large." Velocity tells you "this transaction is large *relative to this card's normal behavior*" — a far stronger fraud signal.

![Velocity KDE](images/plot_fraud_velocity_kde.png)

### Temporal Features

![Fraud by Hour](images/plot_fraud_by_hour.png)

- Extracted `Day`, `Month`, `Hour` from transaction datetime
- Created `is_night` binary flag (hours 0–4 and 21–23) based on observed fraud concentration

### High-Cardinality Features (Target Encoding with Smoothing)
Raw fraud rates from small-sample groups are unreliable — a job category with 9 transactions all fraud yields 100% rate, but it is just noise.

| Layer | Effect |
|---|---|
| Threshold (`min_samples=50`) | Groups with too few samples → use global mean |
| Smoothing (`k=100`) | Shrinks estimates toward global mean based on sample size |
| Out-of-Fold (`GroupKFold`, 5 folds by `cc_num`) | Prevents target leakage from a row's own label |

Tested on: `state`, `city`, `job`, `merchant` — but `state_fraud_rate`, `city_fraud_rate`, and `job_fraud_rate` showed negligible separation between fraud and non-fraud (diff ≈ 0.000007–0.07) and were dropped. Only **`merchant_fraud_rate`** was retained in the final feature set.

### Category Features
- `is_online` — online (`_net`) vs in-person (`_pos`) transactions
- `risk_tier` — High / Mid / Low based on category fraud rate (derived from training set only)

### Age & Amount Features
- Customer age computed from DOB using 2021 as reference year, binned into 6 groups
- `amt` binned into Low / Mid / High to capture the bimodal fraud distribution (~$100 and ~$500 peaks)

---

## ⚖️ Handling Class Imbalance

Three strategies were trained and compared on the (never-resampled) test set:

| Strategy | Approach |
|---|---|
| Original | Use `scale_pos_weight` to upweight fraud class |
| Undersampling | `RandomUnderSampler` to balance classes |
| Oversampling | `SMOTE` with k=5 neighbors to synthesize fraud samples |

**Winner: Original + `scale_pos_weight`** — highest PR-AUC on the real-world imbalanced test set across all models.

---

## ⚙️ Preprocessing Pipeline

```
Raw Data
   ↓ EDA with fraud rate by feature
   ↓ Temporal / age / amount feature engineering
   ↓ Velocity feature engineering (rolling windows per card)
   ↓ Target encoding with smoothing (merchant; state/city/job tested & dropped)
   ↓ Category encoding (is_online, risk_tier)
   ↓ Outlier-preserving scaling (log1p + RobustScaler)
   ↓ Class imbalance handling (3 strategies)
   ↓ Model comparison → Threshold tuning → Business Case Analysis
```

---

## 🤖 Model Comparison

| Model | Strategy | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | Original | 0.4038 | 0.11 | 0.96 | 0.19 |
| Decision Tree | Original | 0.8219 | 0.57 | 0.89 | 0.70 |
| Random Forest | Original | 0.9615 | 0.95 | 0.87 | 0.91 |
| **XGBoost** | **Original** | **0.9750** | 0.89 | 0.95 | **0.92** |
| LightGBM | Original | 0.9748 | 0.88 | 0.95 | 0.92 |

**Winner: XGBoost (Original)** — PR-AUC 0.9750, F1 0.92

---

## 📊 Evaluation

Primary metric: **PR-AUC** (Area Under the Precision-Recall Curve)

PR-AUC is preferred over ROC-AUC for imbalanced datasets because it directly measures performance on the minority (fraud) class, without being inflated by the large number of true negatives.

---

## 📈 Business Case Analysis

**Test period:** Jul–Dec 2020 (~193 days) · 555,719 transactions · 2,145 frauds  
**Baseline (no model):** ~USD 1.14M loss per 6 months

> ⚠️ Cost figures below are illustrative placeholders only (USD 530 avg fraud amount, USD 5 review cost, USD 20 false-block cost). Replace with real institution-specific costs before using for actual deployment decisions.

| Scenario | Threshold | Caught (TP) | Missed (FN) | False Alarms (FP) | Fraud Prevented | Fraud Lost |
|---|---|---|---|---|---|---|
| A. Auto-block | 0.992 | 1,772 (83%) | 373 | 26 | ~USD 939K | ~USD 198K |
| B. Balanced | 0.907 | 1,940 (90%) | 205 | 122 | ~USD 1.03M | ~USD 109K |
| C. Review queue | 0.014 | 2,102 (98%) | 43 | 2,966 | ~USD 1.11M | ~USD 23K |

### Recommended Deployment: Two-Threshold System

- **score ≥ 0.992 → block automatically** (only 26 false blocks in 6 months)
- **0.014 ≤ score < 0.992 → send to review queue** (~17 reviews/day, recovering USD 175K additional fraud)
- **score < 0.014 → approve**

---

## 🚀 How to Run

1. Download the dataset from Kaggle and place CSVs in the input path
2. Open the notebook in Kaggle or Jupyter
3. Run all cells in order — feature engineering → encoding → scaling → model comparison → threshold tuning → business case

```python
# Key dependencies
pip install -r requirements.txt
```

---

## 🔮 Future Improvements

- **Geo-velocity features** — flag physically impossible transactions (e.g. two transactions in different countries within minutes)
- **Hyperparameter Tuning** — current XGBoost uses near-default parameters; tuning via Optuna could improve PR-AUC further
- **Feature Importance Analysis** — evaluate how velocity features rank against target-encoded features to validate engineering effort
- **Online learning** — update model incrementally as new labeled fraud data arrives to adapt to evolving fraud patterns
