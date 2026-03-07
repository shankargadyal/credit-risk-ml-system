import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk & Loan Pre-Eligibility System",
    page_icon="🏦",
    layout="wide"
)

# ── Load models ───────────────────────────────────────────────
@st.cache_resource
def load_models():
    xgb   = joblib.load("models/xgb_model.pkl")
    lr    = joblib.load("models/lr_model.pkl")
    feats = joblib.load("models/feature_names.pkl")
    return xgb, lr, feats

try:
    xgb_model, lr_model, feature_names = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.error(f"❌ Model loading failed: {e}")
    st.stop()

# ── Helpers ───────────────────────────────────────────────────
def risk_band(prob):
    if prob < 0.30:   return "Low Risk",    "#2ecc71"
    elif prob < 0.60: return "Medium Risk", "#f39c12"
    else:             return "High Risk",   "#e74c3c"

def pre_eligibility_check(dti, annual_inc, loan_amnt):
    if dti > 40:
        return "Reject - High DTI", "Debt-to-Income ratio exceeds 40%"
    if annual_inc < 30000:
        return "Reject - Low Income", "Annual income below $30,000"
    if loan_amnt > 0.5 * annual_inc:
        return "Reject - Loan Too Large", "Loan amount exceeds 50% of annual income"
    return "Eligible", ""

def build_input(loan_amnt, term, int_rate, grade, emp_length, annual_inc, dti):
    row = {col: 0 for col in feature_names}
    row['loan_amnt']  = loan_amnt
    row['term']       = term
    row['int_rate']   = int_rate
    row['grade']      = grade
    row['emp_length'] = emp_length
    row['annual_inc'] = annual_inc
    row['dti']        = dti
    return pd.DataFrame([row])[feature_names]

def gauge_chart(prob):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={'suffix': "%", 'font': {'size': 32}},
        title={'text': "Default Probability", 'font': {'size': 16}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar':  {'color': "#e74c3c" if prob > 0.6 else "#f39c12" if prob > 0.3 else "#2ecc71"},
            'steps': [
                {'range': [0, 30],  'color': "rgba(46,204,113,0.15)"},
                {'range': [30, 60], 'color': "rgba(243,156,18,0.15)"},
                {'range': [60, 100],'color': "rgba(231,76,60,0.15)"},
            ],
            'threshold': {
                'line': {'color': "black", 'width': 3},
                'thickness': 0.75,
                'value': prob * 100
            }
        }
    ))
    fig.update_layout(height=260, margin=dict(t=40, b=10, l=20, r=20))
    return fig

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank.png", width=60)
    st.title("🏦 Credit Risk System")
    st.caption("Intelligent Loan Pre-Eligibility using ML")
    st.divider()

    st.subheader("⚙️ Model Selection")
    model_choice = st.selectbox(
        "Choose Model",
        ["XGBoost (Recommended)", "Logistic Regression"]
    )

    st.divider()
    st.subheader("📊 Model Performance")
    st.metric("XGBoost ROC-AUC",  "0.737")
    st.metric("LR ROC-AUC",       "0.735")
    st.metric("Training Samples", "212,620")
    st.metric("Default Rate",     "20.1%")

    st.divider()
    st.caption("Built by: Your Name | MSc Data Science")
    st.caption("Dataset: LendingClub 2007–2018")

# ── Main UI ───────────────────────────────────────────────────
st.title("🏦 Intelligent Credit Risk & Loan Pre-Eligibility System")
st.markdown("*Predict loan default probability using a Hybrid Rule-Based + ML system trained on 265,000+ real loans*")
st.divider()

tab1, tab2, tab3 = st.tabs(["🔍 Loan Eligibility Check", "📊 Model Insights", "ℹ️ About"])

# ═══════════════════════════════════════════════════
# TAB 1 — LOAN ELIGIBILITY CHECK
# ═══════════════════════════════════════════════════
with tab1:
    st.subheader("Enter Applicant Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        loan_amnt = st.number_input(
            "💰 Loan Amount ($)", min_value=500, max_value=500000,
            value=15000, step=500
        )
        annual_inc = st.number_input(
            "💼 Annual Income ($)", min_value=1000, max_value=10000000,
            value=75000, step=1000
        )
        dti = st.slider(
            "📉 Debt-to-Income Ratio (%)", min_value=0.0,
            max_value=60.0, value=18.0, step=0.5
        )

    with col2:
        term = st.selectbox("📅 Loan Term (months)", [36, 60])
        int_rate = st.slider(
            "📈 Interest Rate (%)", min_value=5.0,
            max_value=30.0, value=12.5, step=0.1
        )
        emp_length = st.slider(
            "👔 Employment Length (years)", min_value=0,
            max_value=10, value=5
        )

    with col3:
        grade_label = st.selectbox(
            "🏷️ Loan Grade",
            ["A (Best)", "B", "C", "D", "E", "F", "G (Worst)"]
        )
        grade = {"A (Best)":1,"B":2,"C":3,"D":4,"E":5,"F":6,"G (Worst)":7}[grade_label]

        home_ownership = st.selectbox(
            "🏠 Home Ownership",
            ["MORTGAGE", "RENT", "OWN"]
        )
        purpose = st.selectbox(
            "🎯 Loan Purpose",
            ["debt_consolidation", "credit_card", "home_improvement",
             "major_purchase", "medical", "small_business", "other"]
        )

    st.divider()
    predict_btn = st.button("🔮 Check Eligibility", type="primary", use_container_width=True)

    if predict_btn:
        # Rule check
        rule_result, rule_reason = pre_eligibility_check(dti, annual_inc, loan_amnt)

        if rule_result != "Eligible":
            st.error(f"### ❌ {rule_result}")
            st.warning(f"**Reason:** {rule_reason}")
            st.info("💡 **Recommendation:** Improve your financial profile before reapplying. "
                    "Consider reducing existing debts or increasing your income.")

        else:
            # ML prediction
            input_df = build_input(loan_amnt, term, int_rate, grade, emp_length, annual_inc, dti)
            model = xgb_model if "XGBoost" in model_choice else lr_model

            try:
                prob = model.predict_proba(input_df)[0][1]
                band, color = risk_band(prob)

                # Decision
                if band == "High Risk":
                    decision = "Rejected (High ML Risk)"
                    icon = "❌"
                elif band == "Medium Risk":
                    decision = "Conditional Approval"
                    icon = "⚠️"
                else:
                    decision = "Approved"
                    icon = "✅"

                # Results layout
                r1, r2, r3 = st.columns([1, 1, 1])

                with r1:
                    st.plotly_chart(gauge_chart(prob), use_container_width=True)

                with r2:
                    st.markdown(f"### {icon} Decision")
                    st.markdown(
                        f"<div style='background:{color};padding:16px;border-radius:10px;"
                        f"color:white;font-size:20px;font-weight:bold;text-align:center'>"
                        f"{decision}</div>",
                        unsafe_allow_html=True
                    )
                    st.markdown("")
                    st.markdown(f"**Risk Category:** `{band}`")
                    st.markdown(f"**Default Probability:** `{prob:.1%}`")
                    st.markdown(f"**Model Used:** `{model_choice.split('(')[0].strip()}`")

                with r3:
                    st.markdown("### 📋 Rule Check")
                    checks = [
                        ("DTI ≤ 40%",              dti <= 40),
                        ("Income ≥ $30K",           annual_inc >= 30000),
                        ("Loan ≤ 50% of income",   loan_amnt <= 0.5 * annual_inc),
                    ]
                    for label, passed in checks:
                        st.markdown(f"{'✅' if passed else '❌'} {label}")

                    st.divider()
                    st.markdown("### 💡 Key Risk Factors")
                    if int_rate > 13:
                        st.markdown("⚠️ High interest rate increases risk")
                    if term == 60:
                        st.markdown("⚠️ 60-month term has higher default rate")
                    if dti > 25:
                        st.markdown("⚠️ High debt-to-income ratio")
                    if annual_inc < 50000:
                        st.markdown("⚠️ Income below median borrower")
                    if grade >= 5:
                        st.markdown("⚠️ Grade E–G indicates high borrower risk")
                    if prob < 0.3:
                        st.markdown("✅ Low predicted default probability")

            except Exception as e:
                st.error(f"Prediction error: {e}")

# ═══════════════════════════════════════════════════
# TAB 2 — MODEL INSIGHTS
# ═══════════════════════════════════════════════════
with tab2:
    st.subheader("📊 Model Performance & Risk Analysis")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("XGBoost AUC",  "0.737", "+0.002 vs LR")
    c2.metric("LR AUC",       "0.735")
    c3.metric("RF AUC",       "0.700", "-0.037 vs XGB")
    c4.metric("Training Size", "212K rows")

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Default Rate by Risk Band")
        bands_df = pd.DataFrame({
            "Risk Band":    ["Low Risk", "Medium Risk", "High Risk"],
            "Default Rate": [0.0597,     0.1738,        0.4094],
            "Count":        [13925,      26281,          12950],
        })
        fig = px.bar(
            bands_df, x="Risk Band", y="Default Rate",
            color="Risk Band",
            color_discrete_map={
                "Low Risk":    "#2ecc71",
                "Medium Risk": "#f39c12",
                "High Risk":   "#e74c3c"
            },
            text=bands_df["Default Rate"].apply(lambda x: f"{x:.1%}"),
            title="Actual Default Rate by ML Risk Band"
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("#### Final Decision Distribution")
        dec_df = pd.DataFrame({
            "Decision":     ["Approved", "Conditional", "Rejected ML", "Rejected Rule"],
            "Count":        [13616,       24412,          12079,          3049],
            "Default Rate": [0.0587,      0.1727,         0.4129,         0.2283],
        })
        fig2 = px.bar(
            dec_df, x="Decision", y="Count",
            color="Decision",
            color_discrete_map={
                "Approved":       "#2ecc71",
                "Conditional":    "#f39c12",
                "Rejected ML":    "#e74c3c",
                "Rejected Rule":  "#c0392b"
            },
            title="Hybrid Decision System — Test Set (53K loans)"
        )
        fig2.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.markdown("#### 📋 Model Comparison")
    results_df = pd.DataFrame({
        "Model":             ["Logistic Regression", "XGBoost", "Random Forest"],
        "Accuracy":          [0.6879, 0.6756, 0.7975],
        "ROC-AUC":           [0.7347, 0.7366, 0.7005],
        "Recall (Default)":  [0.64,   0.67,   0.15],
        "Used in App":       ["No", "✅ Yes", "No (729MB)"],
    })
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    st.info(
        "⚠️ **Feature Leakage Note:** `int_rate` and `grade` are strong predictors "
        "but are typically assigned *after* credit decisions in real lending systems. "
        "A production model would exclude them. This project uses them to demonstrate "
        "the full ML pipeline."
    )

# ═══════════════════════════════════════════════════
# TAB 3 — ABOUT
# ═══════════════════════════════════════════════════
with tab3:
    st.subheader("ℹ️ About This Project")

    st.markdown("""
    ### 🏦 Intelligent Credit Risk & Loan Pre-Eligibility System

    This end-to-end ML system predicts whether a loan applicant is likely to default,
    using a **hybrid approach** combining hard business rules with machine learning.

    ---

    ### 🔧 Technical Stack
    | Component | Technology |
    |---|---|
    | ML Models | XGBoost, Logistic Regression, Random Forest |
    | Preprocessing | sklearn Pipeline (Imputer + Scaler) |
    | Class Imbalance | scale_pos_weight (≈ 3.97) |
    | Explainability | SHAP (TreeExplainer) |
    | Frontend | Streamlit + Plotly |
    | Deployment | Streamlit Cloud |

    ---

    ### 📊 Dataset
    - **Source:** LendingClub Accepted Loans 2007–2018
    - **Size:** 265,776 loans (after filtering to Fully Paid / Charged Off)
    - **Features:** 22 (after encoding)
    - **Default Rate:** 20.13%

    ---

    ### 👤 Author
    **Your Name** | MSc Data Science  
    [LinkedIn](#) | [GitHub](#) | [Email](#)
    """)
