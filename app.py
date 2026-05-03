import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

st.set_page_config(page_title="Churn Dashboard", page_icon="🏦", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    min-height: 100vh;
}

[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.08) !important;
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border-right: 1px solid rgba(255,255,255,0.2);
}
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] .stSlider > div > div { background: rgba(255,255,255,0.3) !important; }

[data-testid="stHeader"] { background: transparent; }

.metric-card {
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.35);
    border-radius: 20px;
    padding: 24px 16px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.15), inset 0 1px 0 rgba(255,255,255,0.4);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    animation: fadeSlideUp 0.6s ease forwards;
}
.metric-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 20px 40px rgba(0,0,0,0.2);
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: white !important;
    -webkit-text-fill-color: white !important;
}
.metric-label {
    font-size: 0.75rem;
    color: rgba(255,255,255,0.75) !important;
    margin-top: 6px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.metric-icon { font-size: 1.6rem; margin-bottom: 8px; }

.card {
    background: rgba(255,255,255,0.12);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 20px;
    padding: 24px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.15), inset 0 1px 0 rgba(255,255,255,0.3);
    margin-bottom: 20px;
    animation: fadeSlideUp 0.7s ease forwards;
    transition: box-shadow 0.3s ease, transform 0.3s ease;
}
.card:hover {
    transform: translateY(-3px);
    box-shadow: 0 16px 40px rgba(0,0,0,0.2);
}
.card h3 {
    color: white !important;
    font-size: 1.05rem;
    font-weight: 600;
    margin: 0 0 16px 0;
}

.result-high {
    background: rgba(224,49,49,0.25);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,100,100,0.4);
    border-left: 5px solid #ff6b6b;
    border-radius: 14px;
    padding: 20px;
    animation: pulse-red 2s infinite;
}
.result-high h3 { color: #ff6b6b !important; margin: 0; }
.result-high h2 { color: #ff6b6b !important; margin: 4px 0 0; }
.result-high p  { color: rgba(255,255,255,0.8) !important; margin: 6px 0; }

.result-low {
    background: rgba(45,158,96,0.25);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(100,220,150,0.4);
    border-left: 5px solid #69db7c;
    border-radius: 14px;
    padding: 20px;
}
.result-low h3 { color: #69db7c !important; margin: 0; }
.result-low h2 { color: #69db7c !important; margin: 4px 0 0; }
.result-low p  { color: rgba(255,255,255,0.8) !important; margin: 6px 0; }

.page-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: white !important;
    margin-bottom: 4px;
    text-shadow: 0 2px 10px rgba(0,0,0,0.2);
}
.page-subtitle {
    color: rgba(255,255,255,0.75) !important;
    font-size: 0.95rem;
    margin-bottom: 24px;
}

.risk-item {
    padding: 10px 14px;
    border-radius: 10px;
    margin: 6px 0;
    font-size: 0.88rem;
    color: white !important;
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    transition: background 0.2s ease;
}
.risk-item:hover { background: rgba(255,255,255,0.22); }

.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
    margin: 16px 0;
}

[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] span {
    color: white !important;
}

.stProgress > div > div {
    background: linear-gradient(90deg, #69db7c, #f5a623, #ff6b6b) !important;
    border-radius: 10px;
}

@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse-red {
    0%,100% { box-shadow: 0 0 0 0 rgba(255,107,107,0.3); }
    50%      { box-shadow: 0 0 0 10px rgba(255,107,107,0); }
}
</style>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model    = pickle.load(open("models/churn_model.pkl", "rb"))
    metadata = pickle.load(open("models/metadata.pkl",    "rb"))
    return model, metadata

model, metadata = load_model()

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
st.markdown('<p class="page-subtitle">Real-time churn risk prediction • Powered by Machine Learning</p>', unsafe_allow_html=True)

# ── Metric cards ───────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
risk_color = "#ff6b6b" if churned else "#69db7c"

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
    # Gauge chart
    st.markdown('<div class="card"><h3>📊 Churn Probability Gauge</h3>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    categories = ["Very Low\n0–25%", "Low\n25–50%", "Medium\n50–75%", "High\n75–100%"]
    colors     = ["#2d9e60", "#74c476", "#f5a623", "#e03131"]
    ax.barh(categories, [25,25,25,25], color=colors, edgecolor="white", height=0.5, alpha=0.9)
    ax.axvline(proba*100, color="white", linewidth=3, linestyle="--", label=f"Risk: {proba:.0%}")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Churn Probability (%)", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.legend(loc="lower right", framealpha=0.2, labelcolor="white", facecolor="white")
    plt.tight_layout()
    st.pyplot(fig, transparent=True)
    plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

    # Profile chart
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
    bar_colors = ["#a78bfa" if v > 0.5 else "#f9a8d4" for v in features_display.values()]
    ax2.barh(list(features_display.keys()), list(features_display.values()),
             color=bar_colors, edgecolor="white", alpha=0.9)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Normalised Value", color="white")
    ax2.tick_params(colors="white")
    for spine in ax2.spines.values(): spine.set_visible(False)
    plt.tight_layout()
    st.pyplot(fig2, transparent=True)
    plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    # Prediction result
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

    # Risk factors
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

    # Recommended actions
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