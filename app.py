"""
Intelligent Credit Risk & Loan Pre-Eligibility System
Complete Production-Ready Streamlit App
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CreditIQ — Loan Risk System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --bg:       #0f1117;
    --surface:  #161b27;
    --border:   #222d3d;
    --accent:   #4f8ef7;
    --green:    #22c55e;
    --amber:    #f59e0b;
    --red:      #ef4444;
    --text:     #e2e8f0;
    --muted:    #64748b;
    --sans:     'Inter', sans-serif;
    --mono:     'IBM Plex Mono', monospace;
}

html, body, .stApp {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Cards */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.4rem;
    margin-bottom: 1rem;
}
.card-accent { border-left: 3px solid var(--accent); }
.card-green  { border-left: 3px solid var(--green); }
.card-red    { border-left: 3px solid var(--red); }
.card-amber  { border-left: 3px solid var(--amber); }

/* Metric boxes */
.metric-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem;
    text-align: center;
}
.metric-label { font-size: 0.68rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 6px; font-family: var(--mono); }
.metric-value { font-size: 1.9rem; font-weight: 700; line-height: 1; }
.metric-sub   { font-size: 0.68rem; color: var(--muted); margin-top: 4px; font-family: var(--mono); }
.metric-blue   { color: var(--accent); }
.metric-green  { color: var(--green); }
.metric-amber  { color: var(--amber); }
.metric-red    { color: var(--red); }
.metric-purple { color: #a78bfa; }

/* Badges */
.badge {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 6px;
    font-weight: 600;
    font-size: 0.88rem;
    font-family: var(--sans);
}
.badge-approved    { background: rgba(34,197,94,0.1);  color: var(--green); border: 1px solid rgba(34,197,94,0.25); }
.badge-conditional { background: rgba(245,158,11,0.1); color: var(--amber); border: 1px solid rgba(245,158,11,0.25); }
.badge-rejected    { background: rgba(239,68,68,0.1);  color: var(--red);   border: 1px solid rgba(239,68,68,0.25); }

/* Probability bar */
.prob-bar-container { background: var(--border); border-radius: 4px; height: 6px; width: 100%; margin: 10px 0; }
.prob-bar           { height: 6px; border-radius: 4px; }

/* Page title */
.page-title { font-size: 1.6rem; font-weight: 700; color: var(--text); letter-spacing: -0.5px; margin-bottom: 2px; }
.page-sub   { color: var(--muted); font-size: 0.82rem; margin-bottom: 1.5rem; font-family: var(--mono); }

/* Section label */
.section-label {
    font-family: var(--mono);
    font-size: 0.65rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.75rem;
}

/* Inputs */
.stTextInput>div>div>input,
.stNumberInput>div>div>input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
}
.stSelectbox>div>div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* Buttons */
.stButton > button {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: var(--sans) !important;
    transition: opacity 0.15s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

.stFormSubmitButton > button {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    width: 100% !important;
    padding: 0.65rem !important;
}

/* Tags */
.tag {
    display: inline-block;
    padding: 0.2rem 0.65rem;
    border-radius: 4px;
    font-size: 0.72rem;
    font-family: var(--mono);
    font-weight: 500;
    margin-right: 0.35rem;
    margin-bottom: 0.35rem;
    background: rgba(79,142,247,0.1);
    color: var(--accent);
    border: 1px solid rgba(79,142,247,0.2);
}

/* Step */
.step { display: flex; align-items: flex-start; gap: 0.9rem; margin-bottom: 0.9rem; }
.step-num {
    flex-shrink: 0; width: 24px; height: 24px; border-radius: 50%;
    background: var(--accent); display: flex; align-items: center;
    justify-content: center; font-size: 0.7rem; font-weight: 700;
    color: white; font-family: var(--mono); margin-top: 2px;
}
.step-title { font-weight: 600; color: var(--text); font-size: 0.88rem; }
.step-desc  { color: var(--muted); font-size: 0.8rem; margin-top: 1px; }

/* Misc */
hr { border-color: var(--border) !important; }
.stAlert { border-radius: 8px !important; }
code {
    background: rgba(79,142,247,0.08) !important;
    border: 1px solid rgba(79,142,247,0.15) !important;
    border-radius: 4px !important;
    color: var(--accent) !important;
    padding: 0.1rem 0.35rem !important;
    font-family: var(--mono) !important;
    font-size: 0.85em !important;
}
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    required = {
        'xgb':      'xgb_model.pkl',
        'imputer':  'imputer.pkl',
        'scaler':   'scaler.pkl',
        'features': 'feature_names.pkl',
    }
    loaded = {}
    for key, fname in required.items():
        path = Path(fname)
        if not path.exists():
            return {}
        try:
            loaded[key] = joblib.load(path)
        except EOFError:
            st.warning(f"⚠️ `{fname}` is corrupted (EOFError). Re-save from your notebook.", icon="🔧")
            return {}
        except Exception as e:
            st.warning(f"⚠️ Could not load `{fname}`: {e}", icon="🔧")
            return {}
    return loaded

MODELS    = load_models()
DEMO_MODE = len(MODELS) == 0


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    'loan_amnt', 'term', 'int_rate', 'grade', 'emp_length', 'annual_inc', 'dti',
    'home_ownership_MORTGAGE', 'home_ownership_OWN', 'home_ownership_RENT',
    'purpose_credit_card', 'purpose_debt_consolidation', 'purpose_home_improvement',
    'purpose_house', 'purpose_major_purchase', 'purpose_medical', 'purpose_moving',
    'purpose_other', 'purpose_renewable_energy', 'purpose_small_business',
    'purpose_vacation', 'purpose_wedding',
]

HOME_MAP    = {'RENT': 'home_ownership_RENT', 'OWN': 'home_ownership_OWN',
               'MORTGAGE': 'home_ownership_MORTGAGE', 'ANY': None}
PURPOSE_MAP = {
    'Debt Consolidation': 'purpose_debt_consolidation',
    'Credit Card':        'purpose_credit_card',
    'Home Improvement':   'purpose_home_improvement',
    'Small Business':     'purpose_small_business',
    'Major Purchase':     'purpose_major_purchase',
    'Medical':            'purpose_medical',
    'Moving':             'purpose_moving',
    'Vacation':           'purpose_vacation',
    'Wedding':            'purpose_wedding',
    'House':              'purpose_house',
    'Car':                None,
    'Other':              'purpose_other',
}
GRADE_MAP = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}


def build_input_row(loan_amnt, term, int_rate, grade, emp_length,
                    annual_inc, dti, home_ownership, purpose):
    row = {col: 0 for col in FEATURE_COLS}
    row['loan_amnt']  = loan_amnt
    row['term']       = term
    row['int_rate']   = int_rate
    row['grade']      = GRADE_MAP[grade]
    row['emp_length'] = emp_length
    row['annual_inc'] = annual_inc
    row['dti']        = dti

    home_col = HOME_MAP.get(home_ownership)
    if home_col and home_col in row:
        row[home_col] = 1

    purpose_col = PURPOSE_MAP.get(purpose)
    if purpose_col and purpose_col in row:
        row[purpose_col] = 1

    return pd.DataFrame([row])[FEATURE_COLS]


def risk_band(prob):
    if prob < 0.30: return "Low Risk",    "green"
    if prob < 0.60: return "Medium Risk", "amber"
    return              "High Risk",   "red"


def rule_check(loan_amnt, annual_inc, dti):
    if dti > 40:
        return False, "Debt-to-Income ratio exceeds 40%"
    if annual_inc < 30_000:
        return False, "Annual income below $30,000 minimum"
    if loan_amnt > 0.5 * annual_inc:
        return False, "Loan amount exceeds 50% of annual income"
    return True, ""


def predict(input_df, prob_override=None):
    if prob_override is not None:
        return prob_override

    if not DEMO_MODE:
        try:
            xgb      = MODELS['xgb']
            features = MODELS['features']
            df_aligned = input_df.reindex(columns=features, fill_value=0)
            from sklearn.pipeline import Pipeline as SKPipeline
            if isinstance(xgb, SKPipeline):
                return float(xgb.predict_proba(df_aligned)[0][1])
            else:
                imputer = MODELS['imputer']
                scaler  = MODELS['scaler']
                X = df_aligned.values.astype(float)
                X = imputer.transform(X)
                X = scaler.transform(X)
                return float(xgb.predict_proba(X)[0][1])
        except Exception as e:
            import traceback
            st.error(f"Prediction error: {e}")
            st.code(traceback.format_exc())

    # Demo / fallback formula
    row = input_df.iloc[0]
    p = (
        0.05
        + (row['dti'] / 100) * 0.3
        + (row['int_rate'] / 30) * 0.25
        + (row['grade'] / 7) * 0.2
        - min(row['annual_inc'] / 200_000, 1) * 0.15
        + (0.1 if row['term'] == 60 else 0)
    )
    return float(np.clip(p + np.random.normal(0, 0.02), 0.01, 0.99))


def generate_explanation(row_dict, prob):
    reasons = []
    if row_dict['dti'] > 30:       reasons.append(("⚠️", "High DTI",         f"Your DTI of {row_dict['dti']:.1f}% is above the 30% caution threshold"))
    if row_dict['int_rate'] > 13:  reasons.append(("⚠️", "High Interest Rate", f"Rate of {row_dict['int_rate']:.1f}% signals elevated credit risk"))
    if row_dict['term'] == 60:     reasons.append(("⚠️", "Long Term",          "60-month term increases default exposure"))
    if row_dict['annual_inc'] < 50_000:
        reasons.append(("⚠️", "Income",  f"Income of ${row_dict['annual_inc']:,.0f} is below the preferred $50k threshold"))
    if row_dict['grade'] >= 5:
        reasons.append(("⚠️", "Loan Grade", f"Grade {list(GRADE_MAP)[row_dict['grade']-1]} indicates higher lender-assigned risk"))
    if prob < 0.30:   reasons.append(("✅", "Low Default Risk",    "Predicted default probability is within acceptable range"))
    elif prob < 0.60: reasons.append(("⚠️", "Moderate Risk",       "Default probability warrants careful review"))
    else:             reasons.append(("❌", "High Default Risk",    "Default probability is above acceptable threshold"))
    return reasons


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div style="padding:0.8rem 0 1rem 0">
  <div style="font-size:1.3rem;font-weight:700;color:#e2e8f0;letter-spacing:-0.5px">
    📊 CreditIQ
  </div>
  <div style="font-size:0.7rem;color:#64748b;font-family:'IBM Plex Mono',monospace;
              letter-spacing:1px;text-transform:uppercase;margin-top:2px">
    Loan Risk Platform
  </div>
</div>
""", unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "Navigation",
        ["🏠 Home", "🔍 Risk Assessment", "📈 Analytics Dashboard", "🤖 Model Performance", "💬 AI Assistant", "ℹ️ About"],
        label_visibility="collapsed"
    )

    st.divider()

    if DEMO_MODE:
        st.warning("**Demo Mode** — Place `xgb_model.pkl` in the app folder for real predictions.", icon="⚠️")
    else:
        st.success("**Models loaded** — XGBoost · RF · LR", icon="✅")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown('<h1 class="hero-title">Intelligent Credit Risk<br>& Decision System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Production ML pipeline · LendingClub 2007–2018 · MSc Data Science</p>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, label, val, sub, cls in zip(
        [c1,c2,c3,c4,c5],
        ["Dataset","Best AUC","Default Rate","Models","Test Accuracy"],
        ["265K","0.743","20.1%","3","81.2%"],
        ["loans","XGBoost","in dataset","LR · RF · XGB","holdout set"],
        ["metric-blue","metric-green","metric-amber","metric-purple","metric-blue"]
    ):
        with col:
            st.markdown(f"""<div class="metric-box">
<div class="metric-label">{label}</div>
<div class="metric-value {cls}">{val}</div>
<div class="metric-sub">{sub}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns([3, 2], gap="medium")

    with col_l:
        st.markdown("""<div class="card card-accent">
<p class="section-label">How It Works</p>
<div class="step"><div class="step-num">1</div><div class="step-content">
  <div class="step-title">Pre-Eligibility Rules</div>
  <div class="step-desc">Hard policy checks — DTI cap, income floor, loan-to-income ratio</div>
</div></div>
<div class="step"><div class="step-num">2</div><div class="step-content">
  <div class="step-title">ML Risk Scoring</div>
  <div class="step-desc">XGBoost predicts probability of default from applicant features</div>
</div></div>
<div class="step"><div class="step-num">3</div><div class="step-content">
  <div class="step-title">Risk Band Assignment</div>
  <div class="step-desc">Low &lt;30% · Medium 30–60% · High &gt;60%</div>
</div></div>
<div class="step"><div class="step-num">4</div><div class="step-content">
  <div class="step-title">Explainable Decision</div>
  <div class="step-desc">Approved / Conditional / Rejected with plain-English reasoning</div>
</div></div>
</div>""", unsafe_allow_html=True)

    with col_r:
        st.markdown("""<div class="card">
<p class="section-label">Quick Start</p>
<div class="step"><div class="step-num">1</div><div class="step-content">
  <div class="step-title">Risk Assessment</div>
  <div class="step-desc">Enter loan & financial details</div>
</div></div>
<div class="step"><div class="step-num">2</div><div class="step-content">
  <div class="step-title">Get Decision</div>
  <div class="step-desc">View probability, risk band, outcome</div>
</div></div>
<div class="step"><div class="step-num">3</div><div class="step-content">
  <div class="step-title">Read Explanation</div>
  <div class="step-desc">Understand what drove the decision</div>
</div></div>
<div class="step"><div class="step-num">4</div><div class="step-content">
  <div class="step-title">Explore Analytics</div>
  <div class="step-desc">Charts, model comparison, dataset stats</div>
</div></div>
</div>""", unsafe_allow_html=True)

        st.markdown("""<div class="card">
<p class="section-label">Tech Stack</p>
<span class="tag">Python</span><span class="tag">XGBoost</span>
<span class="tag">Scikit-Learn</span><span class="tag">Streamlit</span>
<span class="tag">Pandas</span><span class="tag">Matplotlib</span>
</div>""", unsafe_allow_html=True)

elif page == "🔍 Risk Assessment":
    st.markdown('<p class="page-title">Risk Assessment</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Enter applicant details to generate a credit risk decision</p>', unsafe_allow_html=True)

    with st.form("application_form"):
        st.markdown('<p class="section-label">Loan Details</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            loan_amnt = st.number_input("Loan Amount ($)", min_value=500, max_value=40000, value=15000, step=500)
        with c2:
            term = st.selectbox("Loan Term", [36, 60], format_func=lambda x: f"{x} months")
        with c3:
            int_rate = st.number_input("Interest Rate (%)", min_value=5.0, max_value=30.0, value=12.5, step=0.1)

        st.markdown('<p class="section-label" style="margin-top:1rem">Applicant Profile</p>', unsafe_allow_html=True)
        c4, c5, c6 = st.columns(3)
        with c4:
            grade = st.selectbox("Loan Grade", ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
                                 help="A = best credit quality, G = highest risk")
        with c5:
            emp_length = st.slider("Employment Length (years)", 0, 10, 5)
        with c6:
            home_ownership = st.selectbox("Home Ownership", ['RENT', 'MORTGAGE', 'OWN', 'ANY'])

        c7, c8, c9 = st.columns(3)
        with c7:
            annual_inc = st.number_input("Annual Income ($)", min_value=5000, max_value=500000, value=75000, step=1000)
        with c8:
            dti = st.number_input("Debt-to-Income Ratio (%)", min_value=0.0, max_value=60.0, value=18.0, step=0.1)
        with c9:
            purpose = st.selectbox("Loan Purpose", list(PURPOSE_MAP.keys()))

        submitted = st.form_submit_button("🔍 Analyse Application", use_container_width=True)

    if submitted:
        eligible, rule_reason = rule_check(loan_amnt, annual_inc, dti)

        if not eligible:
            st.markdown(f"""
<div class="card" style="border-left:3px solid var(--red)">
<span class="badge badge-rejected">❌ REJECTED — Policy Rule</span>
<p style="margin-top:1rem;color:#94a3b8">{rule_reason}</p>
<p style="color:#64748b;font-size:0.85rem">The application was rejected before ML scoring due to a hard policy violation.</p>
</div>
""", unsafe_allow_html=True)

        else:
            input_df = build_input_row(loan_amnt, term, int_rate, grade, emp_length,
                                       annual_inc, dti, home_ownership, purpose)
            prob = predict(input_df)
            band, color = risk_band(prob)

            if band == "Low Risk":
                decision, badge_class, icon = "Approved", "badge-approved", "✅"
            elif band == "Medium Risk":
                decision, badge_class, icon = "Conditional Approval", "badge-conditional", "⚠️"
            else:
                decision, badge_class, icon = "Rejected — High ML Risk", "badge-rejected", "❌"

            # ── Save to session state for AI Assistant context ──
            st.session_state['last_assessment'] = {
                'loan_amnt': loan_amnt,
                'annual_inc': annual_inc,
                'dti': dti,
                'grade': grade,
                'int_rate': int_rate,
                'term': term,
                'purpose': purpose,
                'home_ownership': home_ownership,
                'emp_length': emp_length,
                'prob': prob,
                'band': band,
                'decision': decision,
            }

            col_res, col_prob = st.columns([2, 1])
            with col_res:
                st.markdown(f"""
<div class="card card-accent">
<p class="section-label">Final Decision</p>
<span class="badge {badge_class}">{icon} {decision}</span>
<p style="margin-top:1rem;color:#94a3b8">Risk Category: <b style="color:var(--{color})">{band}</b></p>
</div>""", unsafe_allow_html=True)

            with col_prob:
                bar_color = {"green": "#10b981", "amber": "#f59e0b", "red": "#ef4444"}[color]
                st.markdown(f"""
<div class="card" style="text-align:center">
<p class="section-label">Default Probability</p>
<div class="metric-value metric-{color}">{prob:.1%}</div>
<div class="prob-bar-container">
  <div class="prob-bar" style="width:{prob*100:.1f}%;background:{bar_color}"></div>
</div>
<p style="color:#64748b;font-size:0.75rem">0% &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 100%</p>
</div>""", unsafe_allow_html=True)

            explanations = generate_explanation({
                'dti': dti, 'int_rate': int_rate, 'term': term,
                'annual_inc': annual_inc, 'grade': GRADE_MAP[grade]
            }, prob)

            st.markdown('<div class="card"><p class="section-label">Risk Factors & Explanation</p>', unsafe_allow_html=True)
            for emoji, title, detail in explanations:
                icon_color = {"✅": "var(--green)", "⚠️": "var(--amber)", "❌": "var(--red)"}[emoji]
                st.markdown(f"""
<div style="display:flex;align-items:flex-start;gap:0.75rem;margin-bottom:0.75rem;padding:0.75rem;background:#0a0e1a;border-radius:8px">
  <span style="font-size:1.1rem">{emoji}</span>
  <div>
    <b style="color:{icon_color}">{title}</b>
    <p style="margin:0;color:#94a3b8;font-size:0.85rem">{detail}</p>
  </div>
</div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<p class="section-label" style="margin-top:1rem">Application Summary</p>', unsafe_allow_html=True)
            cols = st.columns(5)
            summary = [
                ("Loan Amount",    f"${loan_amnt:,}",     "blue"),
                ("Annual Income",  f"${annual_inc:,}",    "green"),
                ("DTI Ratio",      f"{dti:.1f}%",         "amber" if dti > 30 else "green"),
                ("Interest Rate",  f"{int_rate:.1f}%",    "amber" if int_rate > 13 else "green"),
                ("Loan Grade",     grade,                  "red" if GRADE_MAP[grade] >= 5 else "green"),
            ]
            for col, (label, val, c) in zip(cols, summary):
                with col:
                    st.markdown(f"""<div class="metric-box">
<div class="metric-label">{label}</div>
<div class="metric-value metric-{c}" style="font-size:1.4rem">{val}</div>
</div>""", unsafe_allow_html=True)

        if DEMO_MODE:
            st.caption("🔧 Running in demo mode — probabilities are synthetic.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ANALYTICS DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📈 Analytics Dashboard":
    st.markdown('<p class="page-title">Analytics Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Insights from the LendingClub training dataset</p>', unsafe_allow_html=True)

    np.random.seed(42)
    grades_dist   = {'A': 0.282, 'B': 0.291, 'C': 0.208, 'D': 0.116, 'E': 0.066, 'F': 0.027, 'G': 0.010}
    default_grade = {'A': 0.06,  'B': 0.12,  'C': 0.20,  'D': 0.29,  'E': 0.38,  'F': 0.47,  'G': 0.52}
    purposes      = ['Debt Consolidation','Credit Card','Home Improvement','Small Business','Major Purchase','Other']
    purpose_pct   = [0.587, 0.161, 0.073, 0.044, 0.034, 0.101]
    default_purp  = [0.19, 0.17, 0.16, 0.30, 0.16, 0.21]

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="card"><p class="section-label">Default Rate by Loan Grade</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.5), facecolor='#111827')
        ax.set_facecolor('#0a0e1a')
        colors = ['#10b981','#34d399','#fbbf24','#f59e0b','#f87171','#ef4444','#b91c1c']
        bars = ax.bar(default_grade.keys(), default_grade.values(), color=colors, edgecolor='none', width=0.6)
        ax.set_ylabel("Default Rate", color='#64748b')
        ax.set_xlabel("Grade", color='#64748b')
        ax.tick_params(colors='#64748b')
        for spine in ax.spines.values(): spine.set_edgecolor('#1e293b')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax.grid(axis='y', color='#1e293b', linewidth=0.5)
        fig.tight_layout()
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="card"><p class="section-label">Loan Purpose Distribution</p>', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(6, 3.5), facecolor='#111827')
        ax2.set_facecolor('#0a0e1a')
        ax2.barh(purposes, purpose_pct, color='#3b82f6', edgecolor='none')
        ax2.set_xlabel("Proportion", color='#64748b')
        ax2.tick_params(colors='#64748b')
        for spine in ax2.spines.values(): spine.set_edgecolor('#1e293b')
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
        ax2.grid(axis='x', color='#1e293b', linewidth=0.5)
        fig2.tight_layout()
        st.pyplot(fig2)
        st.markdown("</div>", unsafe_allow_html=True)

    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown('<div class="card"><p class="section-label">Default Rate by Purpose</p>', unsafe_allow_html=True)
        fig3, ax3 = plt.subplots(figsize=(6, 3.5), facecolor='#111827')
        ax3.set_facecolor('#0a0e1a')
        bar_cols = ['#ef4444' if v > 0.25 else '#f59e0b' if v > 0.18 else '#10b981' for v in default_purp]
        ax3.barh(purposes, default_purp, color=bar_cols, edgecolor='none')
        ax3.set_xlabel("Default Rate", color='#64748b')
        ax3.tick_params(colors='#64748b')
        for spine in ax3.spines.values(): spine.set_edgecolor('#1e293b')
        ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
        ax3.grid(axis='x', color='#1e293b', linewidth=0.5)
        fig3.tight_layout()
        st.pyplot(fig3)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_d:
        st.markdown('<div class="card"><p class="section-label">Risk Band Distribution</p>', unsafe_allow_html=True)
        fig4, ax4 = plt.subplots(figsize=(6, 3.5), facecolor='#111827')
        ax4.set_facecolor('#0a0e1a')
        bands      = ['Low Risk', 'Medium Risk', 'High Risk']
        band_pct   = [25.2, 51.8, 23.0]
        band_color = ['#10b981', '#f59e0b', '#ef4444']
        ax4.bar(bands, band_pct, color=band_color, edgecolor='none', width=0.5)
        ax4.set_ylabel("% of Test Set", color='#64748b')
        ax4.tick_params(colors='#64748b')
        for spine in ax4.spines.values(): spine.set_edgecolor('#1e293b')
        ax4.grid(axis='y', color='#1e293b', linewidth=0.5)
        fig4.tight_layout()
        st.pyplot(fig4)
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown('<p class="section-label">Dataset Statistics</p>', unsafe_allow_html=True)
    ks = st.columns(5)
    kpis = [
        ("Total Loans",       "265,776",  "blue"),
        ("Fully Paid",        "212,275",  "green"),
        ("Charged Off",       "53,501",   "red"),
        ("Default Rate",      "20.1%",    "amber"),
        ("Avg Loan Amount",   "$14,890",  "blue"),
    ]
    for col, (label, val, c) in zip(ks, kpis):
        with col:
            st.markdown(f"""<div class="metric-box">
<div class="metric-label">{label}</div>
<div class="metric-value metric-{c}" style="font-size:1.3rem">{val}</div>
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🤖 Model Performance":
    st.markdown('<p class="page-title">Model Performance</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Comparison of trained classifiers on the held-out test set</p>', unsafe_allow_html=True)

    model_results = {
        "Logistic Regression": {"roc_auc": 0.734, "precision": 0.350, "recall": 0.640, "accuracy": 0.801, "cv_mean": 0.731, "cv_std": 0.003},
        "XGBoost":             {"roc_auc": 0.743, "precision": 0.398, "recall": 0.589, "accuracy": 0.812, "cv_mean": 0.739, "cv_std": 0.004},
        "Random Forest":       {"roc_auc": 0.701, "precision": 0.484, "recall": 0.156, "accuracy": 0.806, "cv_mean": 0.698, "cv_std": 0.005},
    }

    st.markdown('<p class="section-label">Summary Comparison</p>', unsafe_allow_html=True)
    df_results = pd.DataFrame(model_results).T.reset_index().rename(columns={'index': 'Model'})
    df_results.columns = ['Model', 'ROC-AUC', 'Precision', 'Recall', 'Accuracy', 'CV Mean AUC', 'CV Std']
    df_results = df_results.round(3)
    st.dataframe(
        df_results.style
            .highlight_max(subset=['ROC-AUC', 'Recall', 'CV Mean AUC'], color='rgba(59,130,246,0.2)')
            .format({'ROC-AUC': '{:.3f}', 'Precision': '{:.3f}', 'Recall': '{:.3f}',
                     'Accuracy': '{:.3f}', 'CV Mean AUC': '{:.3f}', 'CV Std': '{:.4f}'}),
        use_container_width=True,
        hide_index=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card"><p class="section-label">ROC-AUC Comparison</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.5), facecolor='#111827')
        ax.set_facecolor('#0a0e1a')
        names = list(model_results.keys())
        aucs  = [model_results[n]['roc_auc'] for n in names]
        colors = ['#3b82f6', '#6366f1', '#10b981']
        bars = ax.bar([n.replace(' ', '\n') for n in names], aucs, color=colors, edgecolor='none', width=0.5)
        for bar, val in zip(bars, aucs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{val:.3f}', ha='center', color='#e2e8f0', fontsize=10, fontweight='bold')
        ax.set_ylim(0.65, 0.78)
        ax.set_ylabel("ROC-AUC", color='#64748b')
        ax.tick_params(colors='#64748b')
        for spine in ax.spines.values(): spine.set_edgecolor('#1e293b')
        ax.grid(axis='y', color='#1e293b', linewidth=0.5)
        ax.axhline(0.5, color='#ef4444', linewidth=1, linestyle='--', label='Random baseline')
        fig.tight_layout()
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card"><p class="section-label">Precision vs Recall Trade-off</p>', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(6, 3.5), facecolor='#111827')
        ax2.set_facecolor('#0a0e1a')
        for (name, res), color in zip(model_results.items(), ['#3b82f6', '#6366f1', '#10b981']):
            ax2.scatter(res['recall'], res['precision'], color=color, s=120, zorder=5, label=name)
            ax2.annotate(name.split()[0], (res['recall'], res['precision']),
                         textcoords='offset points', xytext=(8, 4), color=color, fontsize=9)
        ax2.set_xlabel("Recall (on defaulters)", color='#64748b')
        ax2.set_ylabel("Precision", color='#64748b')
        ax2.tick_params(colors='#64748b')
        for spine in ax2.spines.values(): spine.set_edgecolor('#1e293b')
        ax2.grid(color='#1e293b', linewidth=0.5)
        ax2.set_xlim(0.1, 0.75)
        ax2.set_ylim(0.3, 0.55)
        fig2.tight_layout()
        st.pyplot(fig2)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
<div class="card card-accent">
<p class="section-label">Model Selection Rationale</p>
<p><b style="color:#3b82f6">XGBoost</b> is chosen as the production model — it achieves the highest ROC-AUC (0.743) and best cross-validation stability (±0.004).</p>
<p><b style="color:#94a3b8">Logistic Regression</b> is retained as a fast, interpretable fallback.</p>
<p style="color:#64748b;font-size:0.85rem">⚠️ <b>Note:</b> <code>int_rate</code> and <code>grade</code> may introduce data leakage in real-world deployments.</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: AI ASSISTANT CHATBOT
# ─────────────────────────────────────────────────────────────────────────────
elif page == "💬 AI Assistant":
    st.markdown('<p class="page-title">AI Loan Assistant</p>',unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Ask me anything about your loan application, credit risk, or how the system works</p>',unsafe_allow_html=True)

    # ── Gemini API config ──
    GEMINI_API_KEY = "AIzaSyDdTLPDJKrQWW9S20vXBYsF44kGMFYKoWk"
    GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

    KB = {
        "dti": "📊 **Debt-to-Income Ratio (DTI)**\n\nDTI = (Total Monthly Debt ÷ Gross Monthly Income) × 100\n\n• DTI > 40% → Automatic rejection\n• DTI > 30% → Elevated risk flag\n• DTI < 20% → Healthy\n\n**Tip:** Pay down credit cards or car loans before applying.",
        "debt to income": "📊 **DTI** measures how much of your income goes to debt payments.\n\n• >40% → Auto rejected\n• 30–40% → High risk\n• <20% → Healthy",
        "interest rate": "💰 **Interest Rate**\n\n• >13% → Elevated risk flag\n• Grade A: ~6–8%\n• Grade B: ~9–12%\n• Grade C–G: 13–25%+",
        "grade": "🏆 **Loan Grade** (A = best, G = worst)\n\n• A/B → Low risk, approved likely\n• C → Moderate risk\n• D/E → High risk flag\n• F/G → Very likely rejected",
        "approved": "✅ **Approved** — Passed rule check (DTI ≤ 40%, income ≥ $30k, loan ≤ 50% income) AND ML score < 30% default probability.",
        "rejected": "❌ **Rejected** — Either failed a policy rule (DTI>40%, income<$30k, loan>50% income) OR XGBoost predicted >60% default probability.",
        "conditional": "⚠️ **Conditional Approval** — Passed rules but ML flagged medium risk (30–60%). Needs manual review in real lending.",
        "risk": "🎯 **Risk Bands**\n\n• 🟢 Low (<30%) → Approved\n• 🟡 Medium (30–60%) → Conditional\n• 🔴 High (>60%) → Rejected",
        "probability": "📉 **Default Probability** — XGBoost's estimate of non-repayment chance.\n\nIncreased by: high DTI, high rate, 60-month term, low income, bad grade.",
        "default": "⚠️ **Loan Default** — Borrower stops paying. Dataset: 20.1% default rate. Charged Off = defaulted, Fully Paid = repaid.",
        "xgboost": "🤖 **XGBoost** — Best model. ROC-AUC: 0.743. 200 sequential trees, each correcting the previous one.",
        "random forest": "🌲 **Random Forest** — ROC-AUC: 0.701. High precision but misses many defaulters (low recall: 0.156).",
        "logistic regression": "📈 **Logistic Regression** — ROC-AUC: 0.734. Most interpretable, best recall (0.640). Used as regulatory fallback.",
        "roc": "📊 **ROC-AUC**: 1.0 = perfect, 0.5 = random. Best here: 0.743 (XGBoost). 0.7+ is good for credit risk.",
        "auc": "📊 **AUC 0.743** (XGBoost) — distinguishes defaulters from non-defaulters 74.3% of the time.",
        "loan amount": "💵 System rejects loans > 50% of annual income.\n\n• $60k income → max $30k loan\n• $100k income → max $50k loan",
        "term": "📅 **36 months** = lower total interest, lower risk.\n**60 months** = lower monthly payment, slightly higher default risk.",
        "income": "💼 Minimum: **$30,000** (hard reject below). Preferred: **$50k+**. Include all income sources.",
        "employment": "👔 Longer employment = more stability. Range: 0 (< 1 year) to 10 (10+ years).",
        "purpose": "🎯 **Default rates by purpose:**\n• Small Business ~30% (highest)\n• Debt Consolidation ~19%\n• Credit Card ~17%\n• Home Improvement ~16%",
        "home ownership": "🏠 OWN = lowest risk · MORTGAGE = moderate · RENT = slightly higher risk",
        "improve": "💡 **Improve your application:**\n1. Reduce DTI — pay down debts\n2. Request a smaller loan\n3. Choose 36-month term\n4. Include all income sources\n5. Build credit history",
        "how to": "💡 Go to **Risk Assessment** → fill in details → click Analyse → read your result and explanation.",
        "dataset": "📂 **LendingClub 2007–2018** — 265,776 resolved loans, 20.1% default rate.",
        "feature": "🔧 Features: loan amount, term, rate, grade, employment, income, DTI, home ownership, purpose.",
        "precision": "🎯 Precision — of predicted defaults, how many were real:\n• RF: 0.484 · XGB: 0.398 · LR: 0.350",
        "recall": "📡 Recall — of real defaults, how many caught:\n• LR: 0.640 · XGB: 0.589 · RF: 0.156",
        "hello": "👋 Hi! I'm your CreditIQ AI Assistant. Ask me about DTI, loan grades, decisions, or how the ML models work!",
        "hi":    "👋 Hello! Ask me about DTI, loan grades, approval decisions, or how to improve your application.",
        "help":  "🆘 I can help with:\n• DTI, interest rates, loan grades\n• Approved/rejected/conditional explanations\n• Improvement tips\n• XGBoost, ROC-AUC, precision & recall\n\nJust type your question! 👇",
    }

    def rule_based_response(user_input):
        text = user_input.lower().strip()
        if text in KB: return KB[text]
        for keyword, response in KB.items():
            if keyword in text: return response
        return None

    # ── Gemini AI response ──
    def ai_response(messages):
        import requests

        app_context = ""
        if 'last_assessment' in st.session_state:
            d = st.session_state['last_assessment']
            app_context = (
                f"\n\nUSER'S MOST RECENT APPLICATION:\n"
                f"- Loan: ${d['loan_amnt']:,} | Income: ${d['annual_inc']:,} | DTI: {d['dti']}%\n"
                f"- Grade: {d['grade']} | Rate: {d['int_rate']}% | Term: {d['term']} months\n"
                f"- Purpose: {d['purpose']} | Ownership: {d['home_ownership']}\n"
                f"- ML Default Probability: {d['prob']:.2%} | Band: {d['band']} | Decision: {d['decision']}\n"
                f"Use this to give personalised advice when relevant."
            )

        system_prompt = (
            "You are CreditIQ Assistant — a helpful AI embedded in an intelligent loan credit risk "
            "assessment app (MSc Data Science project).\n\n"
            "Help users understand: DTI, loan grades, interest rates, default probability, risk bands, "
            "approval/rejection reasons, how to improve applications, and how XGBoost/RF/LR models work.\n\n"
            "System: LendingClub 2007–2018 dataset, 265,776 loans, 20.1% default rate. "
            "XGBoost (AUC 0.743), RF (0.701), LR (0.734). "
            "Risk bands: Low <30%, Medium 30–60%, High >60%. "
            "Rules: DTI>40% rejected, income<$30k rejected, loan>50% income rejected.\n\n"
            "Be concise, friendly, practical. Use bullet points. "
            "Do not give specific financial/legal advice."
            + app_context
        )

        # Convert to Gemini format
        gemini_messages = []
        for msg in messages[-10:]:
            role = "user" if msg["role"] == "user" else "model"
            gemini_messages.append({"role": role, "parts": [{"text": msg["content"]}]})

        # Gemini requires conversation to start with a user turn
        if gemini_messages and gemini_messages[0]["role"] == "model":
            gemini_messages = gemini_messages[1:]

        # If empty after trim, skip
        if not gemini_messages:
            return "Please ask me a question!"

        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": gemini_messages,
            "generationConfig": {"maxOutputTokens": 400, "temperature": 0.7},
        }

        try:
            r = requests.post(
                f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
                json=payload, timeout=15,
            )
            if r.status_code == 200:
                return r.json()["candidates"][0]["content"]["parts"][0]["text"]
            elif r.status_code == 400:
                return f"❌ Bad request: {r.json().get('error',{}).get('message','Unknown error')}"
            elif r.status_code == 429:
                return "⏳ Rate limit reached. Please wait a moment and try again."
            else:
                return f"⚠️ API error {r.status_code}: {r.text[:200]}"
        except requests.exceptions.Timeout:
            return "⏱️ Request timed out. Please try again."
        except Exception as e:
            return f"⚠️ Connection error: {str(e)}"

    # ── Session state ──
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{"role":"assistant","content":(
            "👋 Hi! I'm your **CreditIQ AI Loan Assistant** — powered by Google Gemini.\n\n"
            "I can help you with:\n"
            "• 📊 Understanding DTI, interest rates, loan grades\n"
            "• ✅ Why your application was approved / rejected\n"
            "• 💡 How to improve your application\n"
            "• 🤖 How XGBoost and the ML models work\n\n"
            "What would you like to know?"
        )}]

    col_chat, col_side = st.columns([3, 1])

    with col_side:
        st.markdown('<div class="card"><p class="section-label">💬 Quick Questions</p>',unsafe_allow_html=True)
        quick_qs = ["What is DTI?","Why was I rejected?","How to improve my application?",
                    "What is XGBoost?","Explain loan grades","What is default probability?",
                    "How does this system work?","What is ROC-AUC?","Explain precision and recall","What features does the model use?"]
        for q in quick_qs:
            if st.button(q, key=f"qb_{q}", use_container_width=True):
                st.session_state.chat_history.append({"role":"user","content":q})
                resp = rule_based_response(q)
                if not resp:
                    with st.spinner("Thinking..."):
                        resp = ai_response(st.session_state.chat_history)
                st.session_state.chat_history.append({"role":"assistant","content":resp})
                st.rerun()
        st.markdown("</div>",unsafe_allow_html=True)

        # AI status badge
        st.markdown("""<div class="card" style="margin-top:1rem">
<p class="section-label">🤖 AI Mode</p>
<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
  <div style="width:8px;height:8px;border-radius:50%;background:#22c55e;box-shadow:0 0 6px #22c55e"></div>
  <span style="color:#22c55e;font-size:0.8rem;font-weight:600">Gemini 2.0 Flash Active</span>
</div>
<p style="color:#64748b;font-size:0.75rem;margin:0">Powered by Google Gemini API. Ask any question for intelligent responses.</p>
</div>""",unsafe_allow_html=True)

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = [{"role":"assistant","content":"Chat cleared! How can I help you?"}]
            st.rerun()

        st.markdown(f"""<div class="card" style="margin-top:1rem">
<p class="section-label">Session Stats</p>
<p style="color:#64748b;font-size:0.82rem">
Messages: <b style="color:#e2e8f0">{len(st.session_state.chat_history)}</b><br>
Mode: <b style="color:#22c55e">Gemini AI + Rules</b>
</p></div>""",unsafe_allow_html=True)

    with col_chat:
        chat_box = st.container(height=520)
        with chat_box:
            for msg in st.session_state.chat_history:
                is_user    = msg["role"] == "user"
                bg         = "#1e3a5f" if is_user else "#111827"
                align      = "flex-end" if is_user else "flex-start"
                label      = "You" if is_user else "🤖 CreditIQ Assistant"
                label_col  = "#3b82f6" if is_user else "#10b981"
                border_col = "#2563eb" if is_user else "#1e293b"
                st.markdown(f"""<div style="display:flex;justify-content:{align};margin-bottom:1rem">
  <div style="max-width:82%;background:{bg};border:1px solid {border_col};border-radius:14px;padding:1rem 1.2rem;box-shadow:0 2px 8px rgba(0,0,0,0.3)">
    <p style="margin:0 0 6px 0;font-size:0.68rem;color:{label_col};font-weight:700;text-transform:uppercase;letter-spacing:1.5px">{label}</p>
    <p style="margin:0;color:#e2e8f0;font-size:0.88rem;line-height:1.6;white-space:pre-wrap">{msg["content"]}</p>
  </div>
</div>""",unsafe_allow_html=True)

        user_input = st.chat_input("Ask about DTI, loan grades, approval decisions, XGBoost...")
        if user_input:
            st.session_state.chat_history.append({"role":"user","content":user_input})
            response = rule_based_response(user_input)
            if not response:
                with st.spinner("Thinking..."):
                    response = ai_response(st.session_state.chat_history)
            if not response:
                response = "Sorry, I couldn't get a response. Please try again!"
            st.session_state.chat_history.append({"role":"assistant","content":response})
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ABOUT
# ─────────────────────────────────────────────────────────────────────────────
elif page == "ℹ️ About":
    st.markdown('<p class="page-title">About the Project</p>', unsafe_allow_html=True)

    st.markdown("""
<div class="card card-accent">
<h3>Project Overview</h3>
<p>CreditIQ is a multi-model machine learning system designed to predict loan default risk
and support pre-eligibility decisions. It integrates Logistic Regression,
Random Forest, and XGBoost with model benchmarking, explainability, and interactive risk visualization.</p>
</div>
<div class="card">
<h3>Dataset</h3>
<p><b>LendingClub Accepted Loans (2007–2018)</b> — 265,776 loans. Default rate: 20.1%.</p>
<p><a href="https://www.kaggle.com/datasets/wordsforthewise/lending-club" target="_blank"
style="color:#3b82f6;text-decoration:none">🔗 Kaggle Dataset</a></p>
</div>
<div class="card">
<h3>Decision Pipeline</h3>
<ol>
<li><b>Pre-Eligibility Rules</b> — Hard rejections for extreme DTI, low income, oversized loan</li>
<li><b>ML Scoring</b> — XGBoost predicts probability of default</li>
<li><b>Risk Banding</b> — Low (&lt;30%) / Medium (30–60%) / High (&gt;60%)</li>
<li><b>Final Decision</b> — Approved / Conditional / Rejected</li>
<li><b>Explanation</b> — Plain-English risk factors for every decision</li>
</ol>
</div>
<div class="card">
<h3>Limitations and Future Work</h3>
<ul>
<li><b>Feature leakage:</b> int_rate and grade should be excluded in real pre-decision models</li>
<li><b>SHAP values:</b> Integration would provide more rigorous explainability</li>
<li><b>MLOps:</b> Model retraining pipeline and data drift monitoring not yet implemented</li>
</ul>
</div>
""", unsafe_allow_html=True)

    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="metric-box">
<div class="metric-label">Dataset</div>
<div class="metric-value metric-blue">265K</div>
<div class="metric-sub">LendingClub loans</div>
</div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="metric-box">
<div class="metric-label">Best AUC</div>
<div class="metric-value metric-green">0.743</div>
<div class="metric-sub">XGBoost</div>
</div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="metric-box">
<div class="metric-label">Models</div>
<div class="metric-value metric-purple">3</div>
<div class="metric-sub">LR · RF · XGB</div>
</div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("""<div class="card">
<p class="section-label">Developer</p>
<p><b style="color:#e2e8f0">Shankar Gadyal</b> — MSc Data Science</p>
<p><a href="https://www.linkedin.com/in/shankargadyal" target="_blank" style="color:#3b82f6">LinkedIn</a>
&nbsp;|&nbsp;
<a href="https://github.com/shankargadyal" target="_blank" style="color:#3b82f6">GitHub</a>
&nbsp;|&nbsp;
<a href="https://credit-risk-ml-system-sg.streamlit.app" target="_blank" style="color:#3b82f6">Live App</a>
</p>
</div>""", unsafe_allow_html=True)
