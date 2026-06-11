# DecodeLabs_Task02_UdamIlukpotha

# 💳 Project 2: Supervised Learning — Fraud Detection Pipeline
### Real Kaggle Credit Card Fraud Dataset | DecodeLabs Industrial Training — Batch 2026

***

## 📥 Dataset Download

**Dataset:** Credit Card Fraud Detection (Real-world data by ULB Machine Learning Group)

👉 **Download here:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

**Steps to download:**
1. Create a free account at [kaggle.com](https://www.kaggle.com)
2. Visit the link above
3. Click the **Download** button → you'll get `creditcard.csv` (~144MB)
4. Place `creditcard.csv` in the **same folder** as your notebook



***

## Project Overview

This project is **Project 2** of the DecodeLabs Data Science Industrial Training Kit (Batch 2026). The objective is to build a **production-grade, leak-free supervised learning pipeline** that detects fraudulent credit card transactions in a highly imbalanced dataset — mirroring real challenges faced by data scientists at financial institutions.

**Real Dataset Stats:**
- **284,807** total transactions
- **492** fraud cases (**0.172%** of total) — extreme imbalance
- **28** PCA-transformed features (V1–V28) + `Amount` + `Time`
- Source: European cardholders, transactions over 2 days in September 2013

***

## The Core Problem: Class Imbalance

A model that predicts "Legitimate" for every transaction achieves **99.83% accuracy** — while catching **zero fraud**. This project teaches you to discard Accuracy entirely and use metrics that actually matter.

***

## Evaluation Metrics Used

| Metric | Formula | Why It Matters |
|--------|---------|---------------|
| **Precision** | TP ÷ (TP + FP) | When we flag fraud, are we right? Reduces false declines |
| **Recall** | TP ÷ (TP + FN) | Did we catch ALL real fraud? Missed fraud = direct loss |
| **F1-Score** | 2 × (P × R) ÷ (P + R) | Harmonic balance of Precision and Recall |
| **ROC-AUC** | Area under ROC curve | Overall model separation power |
| ~~Accuracy~~ | ~~Correct ÷ Total~~ | ~~Discarded — deeply misleading on imbalanced data~~ |

**Recall is the primary metric** — a missed fraud (False Negative) causes real financial loss.

***

## Expected Real-World Scores (Kaggle Dataset)

| Model | Precision | Recall | F1-Score | ROC-AUC |
|-------|-----------|--------|----------|---------|
| Logistic Regression | 0.85–0.92 | 0.60–0.75 | 0.70–0.82 | 0.92–0.96 |
| Random Forest | 0.90–0.96 | 0.78–0.88 | 0.84–0.92 | 0.96–0.99 |

> ⚠️ If you see 1.0 (perfect scores) on ALL metrics, it means data leakage has occurred — SMOTE was likely applied before the train/test split.

***

## SMOTE — Synthetic Minority Over-Sampling Technique

SMOTE creates brand new synthetic fraud examples by interpolating between existing ones:

```
x_new = x_i + λ × (x_nn − x_i),   where λ ~ Uniform(0, 1)
```

| Approach | Problem |
|----------|---------|
| ❌ Undersampling | Destroys valuable legitimate transaction data |
| ❌ Simple Oversampling | Just copies fraud rows → overfitting |
| ✅ SMOTE | Creates genuinely new fraud examples → robust learning |

### ⚠️ The Data Leakage Rule

**NEVER apply SMOTE before the Train/Test Split.**

```
# CORRECT ORDER:
1. Stratified 80/20 split
2. SMOTE runs ONLY inside imblearn.pipeline.Pipeline (training fold only)
3. Test set stays untouched — reflects real-world 0.172% imbalance
```

***

## Pipeline Architectures

### Pipeline 1: Logistic Regression
```
StandardScaler → SMOTE → LogisticRegression
```
- StandardScaler is **mandatory** — LR uses gradient descent with L2 regularization; unscaled features distort the penalty
- Hyperparameters tuned: `smote__k_neighbors` [1][2], `classifier__C` [0.01, 0.1, 1.0]

### Pipeline 2: Random Forest
```
SMOTE → RandomForestClassifier
```
- **No StandardScaler** — tree splits are ordinal and scale-invariant
- Hyperparameters tuned: `smote__k_neighbors` [1][2], `classifier__max_depth` [10, 20, None]

***

## Why imblearn.pipeline.Pipeline (Not sklearn)

`sklearn.pipeline.Pipeline` breaks with SMOTE — its `transform()` only modifies `X`.
SMOTE needs to modify **both `X` and `y`** simultaneously via `fit_resample()`.
`imblearn.pipeline.Pipeline` natively supports this and isolates SMOTE inside every CV fold.

***

## Known Error & Fix

### ValueError: Feature names must be in the same order as they were in fit

This happens when using the real Kaggle dataset because it has a `Time` column the synthetic dataset didn't have.

**Fix — update Step 3 in the notebook:**
```python
# Drop Time and log-transform Amount
X = df.drop(columns=['Class', 'Time'])
X['Amount'] = np.log1p(X['Amount'])

# After train/test split, enforce column order consistency
X_test = X_test[X_train.columns]   # ← This line fixes the ValueError
```

***

## Zero-Leakage Protocol (Completed)

1. ✅ Accuracy discarded — Precision, Recall, F1, ROC-AUC used
2. ✅ Stratified 80/20 split performed BEFORE any SMOTE or Scaling
3. ✅ SMOTE applied ONLY inside `imblearn.pipeline.Pipeline`
4. ✅ `StandardScaler` inside LR pipeline — never fit on full dataset
5. ✅ `GridSearchCV` with `StratifiedKFold(5)` — no leakage during tuning
6. ✅ Test set reflects real-world 0.172% imbalance — untouched throughout
7. ✅ Two models trained and compared on the same held-out test set

***

## Notebook Structure

**File:** `Project2_FraudDetection_Pipeline.ipynb`

| Step | Content |
|------|---------|
| Step 0 | Imports — `sklearn` + `imblearn` |
| Step 1 | Data loading + class imbalance visualization |
| Step 2 | Accuracy trap demonstration |
| Step 3 | Feature prep — drop `Time`, log-transform `Amount` |
| Step 4 | Stratified 80/20 split + `X_test = X_test[X_train.columns]` |
| Step 5 | Logistic Regression `imblearn` pipeline |
| Step 6 | Random Forest `imblearn` pipeline |
| Step 7 | `GridSearchCV` hyperparameter tuning |
| Step 8 | Confusion matrices + classification reports |
| Step 9 | ROC curve comparison |
| Step 10 | Final model selection + Zero-Leakage summary |

***

## Setup & Installation

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Mac/Linux)
source .venv/bin/activate

# Install dependencies
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
```

> `imbalanced-learn` is required for `imblearn.pipeline.Pipeline` and `SMOTE`.
> Without it the notebook will fail at import.

***

## Skills Demonstrated

- `imblearn.pipeline.Pipeline` — production-safe pipeline with SMOTE
- `imblearn.over_sampling.SMOTE` — synthetic minority oversampling
- `sklearn.linear_model.LogisticRegression` — linear classification
- `sklearn.ensemble.RandomForestClassifier` — ensemble tree classification
- `sklearn.model_selection.GridSearchCV` — hyperparameter tuning
- `sklearn.model_selection.StratifiedKFold` — stratified cross-validation
- `sklearn.metrics` — Precision, Recall, F1, ROC-AUC, confusion matrix
- `matplotlib` + `seaborn` — ROC curves, heatmaps

***

## Tools & Environment

- **Language:** Python 3.14
- **IDE:** VS Code with Jupyter extension
- **Environment:** Virtual environment (`.venv`)

***

## Key Takeaway

> *"In enterprise payment infrastructures, a model that classifies every transaction as legitimate achieves near-perfect accuracy while resulting in catastrophic financial loss. Discard Accuracy. Optimize for Recall, F1, and ROC-AUC. Deploy with precision."*
>
> — DecodeLabs Industrial Training Kit, Batch 2026