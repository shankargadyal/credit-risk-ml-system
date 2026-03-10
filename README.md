# credit-risk-ml-system
Intelligent Credit Risk &amp; Loan Pre-Eligibility System using Machine Learning | Streamlit + XGBoost + LendingClub Dataset

# 🏦 Intelligent Credit Risk & Loan Pre-Eligibility System
> **End-to-end ML system** that predicts loan default probability and makes eligibility decisions using a hybrid Rule-Based + Machine Learning pipeline — trained on 265,000+ real LendingClub loans.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-ROC--AUC%200.74-green?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-red?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## 🔗 Live Demo
**👉 [Try the App Here] https://credit-risk-ml-system-sg.streamlit.app/


![App Screenshot] <img width="1853" height="900" alt="Screenshot 2026-03-08 004214" src="https://github.com/user-attachments/assets/5897d9f7-90de-4fee-bd5f-3cac573675eb" />
<img width="1876" height="971" alt="Screenshot 2026-03-08 004258" src="https://github.com/user-attachments/assets/b933d513-16fe-4137-8a5c-77fcdf9e4b75" />

---

## 📌 Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [System Architecture](#system-architecture)
- [ML Pipeline](#ml-pipeline)
- [Model Results](#model-results)
- [SHAP Explainability](#shap-explainability)
- [Hybrid Decision System](#hybrid-decision-system)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Key Learnings](#key-learnings)
- [Future Improvements](#future-improvements)

---

## 🎯 Problem Statement

Banks and fintech companies lose billions annually to loan defaults. The challenge:

> **Can we predict — before approving a loan — whether a borrower is likely to default?**

This system solves it with a **two-layer approach**:
1. **Rule Engine** — hard filters for obvious rejections (high DTI, low income)
2. **ML Model** — probabilistic risk scoring for nuanced cases

**Real-world relevance:** Companies like Razorpay, CRED, and Zerodha use similar systems to protect their lending books.

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | [LendingClub Accepted Loans 2007–2018](https://www.kaggle.com/datasets/wordsforthewise/lending-club) |
| Raw rows loaded | 300,000 |
| After filtering (Fully Paid / Charged Off only) | **265,776** |
| Features used | 22 (after encoding) |
| Default rate | 20.13% |
| Train / Test split | 80% / 20% (stratified) |

**Features:**
`loan_amnt`, `term`, `int_rate`, `grade`, `emp_length`, `home_ownership`, `annual_inc`, `dti`, `purpose`

> ⚠️ **Note on Feature Leakage:** `int_rate` and `grade` are typically assigned *after* the credit decision in production. This project uses them to demonstrate the full ML pipeline. A production system would exclude them and rely only on applicant-provided data.

---

## 🏗️ System Architecture

```
Applicant Input
      │
      ▼
┌─────────────────────┐
│  Rule-Based Engine  │  ← Hard reject: DTI > 40, Income < 30K, Loan > 50% income
└─────────────────────┘
      │ Eligible
      ▼
┌─────────────────────┐
│   XGBoost Model     │  ← Predict default probability
│   (sklearn Pipeline)│
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│   Risk Banding      │  ← Low (<30%) / Medium (30-60%) / High (>60%)
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  Final Decision     │  Approved / Conditional / Rejected
│  + SHAP Explanation │  ← Why was this decision made?
└─────────────────────┘
```

---

## 🔧 ML Pipeline

### Preprocessing
```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # handles missing emp_length
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_features)
])
```

### Models Trained
```python
# 1. Logistic Regression (baseline)
lr_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=3000, class_weight='balanced'))
])

# 2. XGBoost (best performer)
xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,  # handles class imbalance
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',
        early_stopping_rounds=30,
        random_state=42
    ))
])

# 3. Random Forest (comparison)
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42
    ))
])
```

### Class Imbalance Handling
```python
# 80% Fully Paid vs 20% Charged Off
# Used scale_pos_weight instead of SMOTE to avoid memory issues on 265K rows
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
# Result: scale_pos_weight ≈ 3.97
```

---

## 📈 Model Results

### Comparison Table
| Model | Accuracy | ROC-AUC | Notes |
|---|---|---|---|
| Logistic Regression | 0.6879 | **0.7347** | Fast, interpretable baseline |
| **XGBoost** | 0.6756 | **0.7366** | ✅ Best AUC, used in production |
| Random Forest | 0.7975 | 0.7005 | High accuracy but poor recall on defaults |

### XGBoost Detailed Report
```
              precision  recall  f1-score  support
  Fully Paid      0.89    0.68      0.77    42,456
 Charged Off      0.34    0.67      0.45    10,700
    accuracy                        0.68    53,156
```

### 5-Fold Cross-Validation
| Model | Mean AUC | Std |
|---|---|---|
| Logistic Regression | 0.7292 | ±0.0030 |
| XGBoost | *see fix below* | — |

> ⚠️ **Known Issue:** XGBoost CV failed due to `early_stopping_rounds` requiring `eval_set`. Fixed with a separate CV pipeline (see `notebooks/cv_fix.py`).

### Risk Band Validation
| Risk Band | Applicants | Actual Default Rate |
|---|---|---|
| 🟢 Low Risk (prob < 0.30) | 13,925 | **5.97%** |
| 🟡 Medium Risk (0.30–0.60) | 26,281 | **17.38%** |
| 🔴 High Risk (prob > 0.60) | 12,950 | **40.94%** |

*The risk bands are well-separated — Low Risk defaults at 6x lower rate than High Risk. This validates the model's business utility.*

---

## 🔍 SHAP Explainability

SHAP (SHapley Additive exPlanations) explains **why** each prediction was made — critical for regulatory compliance in real fintech systems.

```python
import shap

# Create SHAP explainer from trained XGBoost
explainer = shap.TreeExplainer(xgb_pipeline.named_steps['classifier'])

# Transform test data through preprocessing step
X_test_transformed = preprocessor.transform(X_test[:500])

# Calculate SHAP values
shap_values = explainer.shap_values(X_test_transformed)

# Global feature importance plot
shap.summary_plot(
    shap_values, 
    X_test_transformed,
    feature_names=X.columns.tolist(),
    plot_type='bar'
)

# Individual prediction explanation (waterfall chart)
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X_test_transformed[0],
        feature_names=X.columns.tolist()
    )
)
```

**Key finding:** `term`, `grade`, and `int_rate` are the top 3 predictors of default — confirming domain intuition that longer loan terms and lower credit grades carry higher risk.

---

## ⚖️ Hybrid Decision System

### Layer 1: Rule Engine (Pre-Eligibility)
```python
def pre_eligibility_check(row):
    """Hard policy rules applied BEFORE ML scoring."""
    if row['dti'] > 40:
        return "Reject - High DTI"
    if row['annual_inc'] < 30000:
        return "Reject - Low Income"
    if row['loan_amnt'] > 0.5 * row['annual_inc']:
        return "Reject - Loan Too Large"
    return "Eligible"
```

### Layer 2: ML Risk Scoring
```python
def risk_band(prob):
    if prob < 0.30: return "Low Risk"
    elif prob < 0.60: return "Medium Risk"
    else: return "High Risk"
```

### Layer 3: Final Decision
```python
def final_decision(row):
    if row['Pre_Eligibility'] != 'Eligible':
        return 'Rejected (Policy Rule)'
    elif row['Risk_Category'] == 'High Risk':
        return 'Rejected (High ML Risk)'
    elif row['Risk_Category'] == 'Medium Risk':
        return 'Conditional Approval'
    else:
        return 'Approved'
```

### Decision Outcomes on Test Set (53,156 loans)
| Decision | Count | Actual Default Rate |
|---|---|---|
| ✅ Approved | 13,616 | **5.87%** |
| ⚠️ Conditional Approval | 24,412 | **17.27%** |
| ❌ Rejected (High ML Risk) | 12,079 | **41.29%** |
| ❌ Rejected (Policy Rule) | 3,049 | 22.83% |

---

## 📁 Project Structure

```
credit-risk-system/
│
├── 📓 notebooks/
│   ├── 01_eda.ipynb                    # Exploratory data analysis
│   ├── 02_modelling.ipynb              # Model training & evaluation
│   └── 03_shap_explainability.ipynb    # SHAP analysis
│
├── 🤖 models/
│   ├── xgb_model.pkl                   # XGBoost pipeline (2.2 MB)
│   ├── lr_model.pkl                    # Logistic Regression pipeline
│   └── feature_names.pkl              # Feature column order
│
├── 🖥️ app/
│   ├── app.py                          # Streamlit web application
│   └── predict.py                      # Prediction + explanation logic
│
├── 📊 assets/
│   ├── app_screenshot.png
│   ├── roc_curve.png
│   ├── shap_summary.png
│   └── architecture_diagram.png
│
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/credit-risk-system.git
cd credit-risk-system
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Streamlit app locally
```bash
streamlit run app/app.py
```

### 4. Or run prediction directly in Python
```python
import joblib, pandas as pd

# Load model
model = joblib.load('models/xgb_model.pkl')
features = joblib.load('models/feature_names.pkl')

# Sample applicant
applicant = {
    'loan_amnt': 15000, 'term': 36, 'int_rate': 12.5,
    'grade': 3, 'emp_length': 5, 'annual_inc': 75000, 'dti': 18
}

# Build input
df = pd.DataFrame([{col: applicant.get(col, 0) for col in features}])

# Predict
prob = model.predict_proba(df)[0][1]
print(f"Default Probability: {prob:.1%}")
```

### requirements.txt
```
pandas==2.1.0
numpy==1.24.0
scikit-learn==1.6.1
xgboost==2.0.0
shap==0.44.0
streamlit==1.32.0
plotly==5.18.0
seaborn==0.13.0
matplotlib==3.8.0
joblib==1.3.2
```

---

## 💡 Key Learnings

**1. Class imbalance is the hardest part**
With 80/20 class split, a naive model just predicts "Fully Paid" for everything and gets 80% accuracy. Real work is in recall — catching actual defaulters. Used `scale_pos_weight` in XGBoost to handle this.

**2. Pipeline > standalone model**
Wrapping preprocessor + model in a `sklearn.Pipeline` prevents data leakage, makes deployment cleaner, and ensures train/test transformations are identical.

**3. Feature leakage is a real production concern**
`int_rate` and `grade` are strong predictors but are typically assigned *after* credit decisions in real systems. A production model would exclude them — accepting lower AUC in exchange for a truly pre-decision model.

**4. Business metrics > accuracy**
80% accuracy sounds great but means nothing if you're missing 60% of actual defaults. Framing results in terms of **default rate per risk band** is what business stakeholders actually care about.

**5. Early stopping + CV conflict**
XGBoost's `early_stopping_rounds` requires an `eval_set` — which `cross_val_score` doesn't provide. Solution: use a separate XGBoost config without early stopping for CV, and the tuned version for final training.

---

## 🔮 Future Improvements

- [ ] Add **SHAP waterfall charts** in Streamlit app for per-applicant explanations
- [ ] Exclude `int_rate` and `grade` to build a true **pre-decision model**
- [ ] Add **SMOTE** or **ADASYN** for oversampling (test on subset first due to RAM)
- [ ] Integrate **FastAPI** REST endpoint for programmatic access
- [ ] Add **model monitoring** — track prediction drift over time
- [ ] Experiment with **LightGBM** (faster than XGBoost, similar performance)
- [ ] Add **Optuna** hyperparameter tuning for XGBoost

---

## 👤 Author

SHANKAR
MSc Data Science | [LinkedIn](https://www.linkedin.com/in/shankargadyal) | [GitHub](https://github.com/yourusername)

---

## 📄 License
MIT License — feel free to use this project for learning and portfolio purposes.

---

*Built with ❤️ using Python, XGBoost, SHAP, and Streamlit*
