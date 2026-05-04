import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

st.set_page_config(page_title="Churn Dashboard", page_icon="🏦", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

[data-testid="stAppViewContainer"] {
    background: linear-gradient(270deg, #ffffff, #f5f5f5, #e8e8e8, #f0f0f0, #ffffff);
    background-size: 400% 400%;
    animation: gradientMove 8s ease infinite;
    min-height: 100vh;
}

@keyframes gradientMove {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.75) !important;
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border-right: 1px solid rgba(0,0,0,0.08);
}
[data-testid="stSidebar"] * { color: #1a1a1a !important; }
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: rgba(255,255,255,0.95) !important;
    color: #1a1a1a !important;
    border: 1px solid rgba(0,0,0,0.12) !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] .stNumberInput > div > div > input {
    background: rgba(255,255,255,0.95) !important;
    color: #1a1a1a !important;
    border: 1px solid rgba(0,0,0,0.12) !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] .stNumberInput button {
    background: rgba(255,255,255,0.95) !important;
    color: #1a1a1a !important;
    border: 1px solid rgba(0,0,0,0.12) !important;
}
[data-testid="stSidebar"] .stNumberInput > div {
    background: rgba(255,255,255,0.95) !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] select,
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] input[type="number"] {
    background: rgba(255,255,255,0.95) !important;
    color: #1a1a1a !important;
}
[data-testid="stSidebar"] .stSlider > div > div > div {
    background: #4c6ef5 !important;
}
.metric-card {
    background: rgba(255,255,255,0.7);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.9);
    border-radius: 20px;
    padding: 24px 16px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.08), inset 0 1px 0 rgba(255,255,255,0.9);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    animation: fadeSlideUp 0.6s ease forwards;
}
.metric-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 20px 40px rgba(0,0,0,0.12);
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #1a1a1a !important;
    -webkit-text-fill-color: #1a1a1a !important;
}
.metric-label {
    font-size: 0.75rem;
    color: #666666 !important;
    margin-top: 6px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.metric-icon { font-size: 1.6rem; margin-bottom: 8px; }
.card {
    background: rgba(255,255,255,0.65);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.9);
    border-radius: 20px;
    padding: 24px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.07), inset 0 1px 0 rgba(255,255,255,0.9);
    margin-bottom: 20px;
    animation: fadeSlideUp 0.7s ease forwards;
    transition: box-shadow 0.3s ease, transform 0.3s ease;
}
.card:hover {
    transform: translateY(-3px);
    box-shadow: 0 16px 40px rgba(0,0,0,0.1);
}
.card h3 {
    color: #1a1a1a !important;
    font-size: 1.05rem;
    font-weight: 600;
    margin: 0 0 16px 0;
}
.result-high {
    background: rgba(224,49,49,0.08);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(224,49,49,0.25);
    border-left: 5px solid #e03131;
    border-radius: 14px;
    padding: 20px;
    animation: pulse-red 2s infinite;
}
.result-high h3 { color: #c92a2a !important; margin: 0; }
.result-high h2 { color: #c92a2a !important; margin: 4px 0 0; }
.result-high p  { color: #555555 !important; margin: 6px 0; }
.result-low {
    background: rgba(45,158,96,0.08);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(45,158,96,0.25);
    border-left: 5px solid #2d9e60;
    border-radius: 14px;
    padding: 20px;
}
.result-low h3 { color: #2b8a3e !important; margin: 0; }
.result-low h2 { color: #2b8a3e !important; margin: 4px 0 0; }
.result-low p  { color: #555555 !important; margin: 6px 0; }
.page-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: #1a1a1a !important;
    margin-bottom: 4px;
}
.page-subtitle {
    color: #666666 !important;
    font-size: 0.95rem;
    margin-bottom: 24px;
}
.risk-item {
    padding: 10px 14px;
    border-radius: 10px;
    margin: 6px 0;
    font-size: 0.88rem;
    color: #1a1a1a !important;
    background: rgba(255,255,255,0.6);
    border: 1px solid rgba(0,0,0,0.07);
    transition: background 0.2s ease;
}
.risk-item:hover { background: rgba(255,255,255,0.95); }
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,0,0,0.12), transparent);
    margin: 16px 0;
}
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] span,
[data-testid="stAppViewContainer"] div { color: #1a1a1a !important; }
.stProgress > div > div {
    background: linear-gradient(90deg, #2d9e60, #f5a623, #e03131) !important;
    border-radius: 10px;
}
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse-red {
    0%,100% { box-shadow: 0 0 0 0 rgba(224,49,49,0.15); }
    50%      { box-shadow: 0 0 0 8px rgba(224,49,49,0); }
}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
CATEGORICAL = ["Geography", "Gender"]
NUMERICAL   = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
    "HasCrCard", "IsActiveMember", "EstimatedSalary",
    "BalanceSalaryRatio", "TenureByAge", "CreditScoreByAge",
    "ZeroBalance", "ProductsPerTenure"
]

# ── Train model from scratch ───────────────────────────────────────────────────
def train_model():
    df = pd.read_csv("data/Churn_Modelling.csv")
    df.drop(columns=["RowNumber", "CustomerId", "Surname"], inplace=True)
    df["BalanceSalaryRatio"] = df["Balance"] / (df["EstimatedSalary"] + 1)
    df["TenureByAge"]        = df["Tenure"] / df["Age"]
    df["CreditScoreByAge"]   = df["CreditScore"] / df["Age"]
    df["ZeroBalance"]        = (df["Balance"] == 0).astype(int)
    df["ProductsPerTenure"]  = df["NumOfProducts"] / (df["Tenure"] + 1)
    X = df[CATEGORICAL + NUMERICAL]
    y = df["Exited"]
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERICAL),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), CATEGORICAL),
    ])
    model = Pipeline([
        ("pre", preprocessor),
        ("clf", GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42))
    ])
    model.fit(X, y)
    os.makedirs("models", exist_ok=True)
    pickle.dump(model, open("models/churn_model.pkl", "wb"))
    return model

# ── Load or retrain model ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open("models/churn_model.pkl", "rb"))
        test  = pd.DataFrame([{
            "Geography": "France", "Gender": "Male", "CreditScore": 650,
            "Age": 40, "Tenure": 5, "Balance": 50000, "NumOfProducts": 1,
            "HasCrCard": 1, "IsActiveMember": 1, "EstimatedSalary": 80000,
            "BalanceSalaryRatio": 0.6, "TenureByAge": 0.1,
            "CreditScoreByAge": 16.0, "ZeroBalance": 0, "ProductsPerTenure": 0.2
        }])
        model.predict(test)
        return model
    except Exception:
        return train_model()

model = load_model()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 ChurnPredict AI")
    st.markdown("*Bank Customer Analytics*")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("### 👤 Customer Info")
    geography = st.selectbox("Country",  ["France", "Germany", "Spain"])
    gender    = st.selectbox("Gender",   ["Male", "Female"])
    age       = st.slider("Age", 18, 92, 40)
    tenure    = st.slider("Years with Bank", 0, 10, 5)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("### 💰 Financial Info")
    credit_score = st.slider("Credit Score", 300, 850, 650)
    balance      = st.number_input("Balance ($)",     0.0, 300000.0, 50000.0, step=1000.0)
    salary       = st.number_input("Est. Salary ($)", 0.0, 300000.0, 80000.0, step=1000.0)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("### 🏦 Account Details")
    num_products = st.selectbox("No. of Products", [1, 2, 3, 4])
    has_cr_card  = st.radio("Has Credit Card?",  ["Yes", "No"])
    is_active    = st.radio("Is Active Member?", ["Yes", "No"])

# ── Input dataframe ────────────────────────────────────────────────────────────
input_df = pd.DataFrame([{
    "Geography":          geography,
    "Gender":             gender,
    "CreditScore":        credit_score,
    "Age":                age,
    "Tenure":             tenure,
    "Balance":            balance,
    "NumOfProducts":      num_products,
    "HasCrCard":          1 if has_cr_card == "Yes" else 0,
    "IsActiveMember":     1 if is_active   == "Yes" else 0,
    "EstimatedSalary":    salary,
    "BalanceSalaryRatio": balance / (salary + 1),
    "TenureByAge":        tenure / age,
    "CreditScoreByAge":   credit_score / age,
    "ZeroBalance":        1 if balance == 0 else 0,
    "ProductsPerTenure":  num_products / (tenure + 1),
}])

proba   = model.predict_proba(input_df)[0][1]
churned = proba >= 0.5

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="page-title">🏦 Customer Churn Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="page-subtitle">Real-time churn risk prediction</p>', unsafe_allow_html=True)

# ── Metric cards ───────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
risk_color = "#c92a2a" if churned else "#2b8a3e"

with c1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-icon">💳</div>
        <div class="metric-value">{credit_score}</div>
        <div class="metric-label">Credit Score</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-icon">💰</div>
        <div class="metric-value">${balance:,.0f}</div>
        <div class="metric-label">Account Balance</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-icon">📅</div>
        <div class="metric-value">{tenure} yrs</div>
        <div class="metric-label">Tenure</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-icon">⚠️</div>
        <div class="metric-value" style="color:{risk_color} !important; -webkit-text-fill-color:{risk_color} !important;">{proba:.0%}</div>
        <div class="metric-label">Churn Risk</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Charts + results ───────────────────────────────────────────────────────────
left, right = st.columns([1.2, 1])

with left:
    st.markdown('<div class="card"><h3>📊 Churn Probability Gauge</h3>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    categories = ["Very Low\n0–25%", "Low\n25–50%", "Medium\n50–75%", "High\n75–100%"]
    colors     = ["#2d9e60", "#74c476", "#f5a623", "#e03131"]
    ax.barh(categories, [25,25,25,25], color=colors, edgecolor="white", height=0.5, alpha=0.85)
    ax.axvline(proba*100, color="#1a1a1a", linewidth=3, linestyle="--", label=f"Risk: {proba:.0%}")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Churn Probability (%)", color="#1a1a1a")
    ax.tick_params(colors="#1a1a1a")
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.legend(loc="lower right", framealpha=0.4, labelcolor="#1a1a1a")
    plt.tight_layout()
    st.pyplot(fig, transparent=True)
    plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><h3>📈 Customer Profile</h3>', unsafe_allow_html=True)
    features_display = {
        "Credit Score": credit_score / 850,
        "Balance":      min(balance / 200000, 1),
        "Tenure":       tenure / 10,
        "Age":          min(age / 80, 1),
        "Products":     num_products / 4,
        "Salary":       min(salary / 200000, 1),
    }
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    fig2.patch.set_alpha(0)
    ax2.set_facecolor("none")
    bar_colors = ["#4c6ef5" if v > 0.5 else "#94d82d" for v in features_display.values()]
    ax2.barh(list(features_display.keys()), list(features_display.values()),
             color=bar_colors, edgecolor="white", alpha=0.85)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Normalised Value", color="#1a1a1a")
    ax2.tick_params(colors="#1a1a1a")
    for spine in ax2.spines.values(): spine.set_visible(False)
    plt.tight_layout()
    st.pyplot(fig2, transparent=True)
    plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card"><h3>🎯 Prediction Result</h3>', unsafe_allow_html=True)
    if churned:
        st.markdown(f"""<div class="result-high">
            <h3>🔴 HIGH CHURN RISK</h3>
            <p>This customer is likely to leave.</p>
            <h2>{proba:.1%} probability</h2>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="result-low">
            <h3>🟢 LOW CHURN RISK</h3>
            <p>This customer is likely to stay.</p>
            <h2>{proba:.1%} probability</h2>
        </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.progress(float(proba))
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><h3>⚠️ Risk Factors</h3>', unsafe_allow_html=True)
    risks = []
    if age > 55:               risks.append(("🔴", "Older customer — higher churn tendency"))
    if balance == 0:           risks.append(("🔴", "Zero balance — disengaged customer"))
    if num_products == 1:      risks.append(("🟡", "Only 1 product — low engagement"))
    if is_active == "No":      risks.append(("🔴", "Inactive member"))
    if tenure < 2:             risks.append(("🟡", "New customer — still settling in"))
    if geography == "Germany": risks.append(("🟡", "Germany has higher churn rates"))
    if credit_score < 500:     risks.append(("🔴", "Low credit score"))
    if not risks:              risks.append(("🟢", "No major risk factors detected"))
    for icon, text in risks:
        st.markdown(f'<div class="risk-item">{icon} {text}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><h3>💡 Recommended Actions</h3>', unsafe_allow_html=True)
    if proba >= 0.7:
        actions = ["🚨 Assign relationship manager immediately",
                   "🎁 Offer personalised retention package",
                   "📞 Schedule a call within 48 hours",
                   "💸 Provide fee waivers or rate benefits"]
    elif proba >= 0.5:
        actions = ["📋 Send a satisfaction survey",
                   "💼 Offer product upgrade or cross-sell",
                   "🔍 Check for recent complaints",
                   "📧 Initiate personalised outreach"]
    else:
        actions = ["⭐ Recommend additional products",
                   "🎖️ Enrol in loyalty rewards program",
                   "💳 Encourage credit card usage",
                   "📊 Share investment or savings options"]
    for action in actions:
        st.markdown(f'<div class="risk-item">{action}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
