import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from fpdf import FPDF
from groq import Groq
from datetime import datetime
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings('ignore')

client = Groq(api_key="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")

st.set_page_config(page_title="FinAI", page_icon="💰", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; }
[data-testid="stAppViewContainer"] { background-color: #080b12; }
[data-testid="stSidebar"] { background-color: #080b12; }
[data-testid="stHeader"] { background: transparent; }

.hero {
    text-align: center;
    padding: 60px 20px 40px;
    background: radial-gradient(ellipse at 50% 0%, rgba(55,138,221,0.15) 0%, transparent 70%);
}
.hero-title {
    font-size: 52px;
    font-weight: 800;
    background: linear-gradient(135deg, #ffffff 0%, #378ADD 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.1;
}
.hero-sub {
    font-size: 16px;
    color: #4a5568;
    margin-top: 12px;
}
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #1e2535, transparent);
    margin: 40px 0;
}
.sec {
    font-size: 13px;
    font-weight: 600;
    color: #378ADD;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin: 40px 0 16px;
}
.card {
    background: #0d1117;
    border-radius: 16px;
    padding: 20px;
    border: 1px solid #1a2035;
    text-align: center;
    transition: border-color 0.2s;
}
.card:hover { border-color: #378ADD44; }
.card-val { font-size: 28px; font-weight: 700; color: #fff; }
.card-lbl { font-size: 11px; color: #4a5568; margin-top: 6px; text-transform: uppercase; letter-spacing: 1px; }
.card-icon { font-size: 22px; margin-bottom: 8px; }

.score-ring-wrap {
    background: #0d1117;
    border-radius: 20px;
    padding: 40px 30px;
    border: 1px solid #1a2035;
    text-align: center;
}
.score-big { font-size: 88px; font-weight: 900; line-height: 1; }
.score-sub { font-size: 13px; color: #4a5568; margin-top: 4px; }
.score-lbl { font-size: 22px; font-weight: 700; margin-top: 10px; }

.goal-card {
    background: #0d1117;
    border-radius: 16px;
    padding: 24px;
    border: 1px solid #1a2035;
    margin-bottom: 16px;
}
.progress-track {
    background: #131720;
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
    margin: 10px 0;
}
.progress-fill {
    height: 100%;
    border-radius: 999px;
}

.budget-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 0;
    border-bottom: 1px solid #1a2035;
}
.budget-row:last-child { border-bottom: none; }

.tip-card {
    background: #0d1420;
    border-radius: 12px;
    padding: 16px;
    border: 1px solid #1a3050;
    margin-bottom: 10px;
}

div[data-testid="stNumberInput"] > div > div > input {
    background: #0d1117 !important;
    border: 1px solid #1a2035 !important;
    border-radius: 10px !important;
    color: #fff !important;
    font-size: 14px !important;
}
div[data-testid="stTextInput"] > div > div > input {
    background: #0d1117 !important;
    border: 1px solid #1a2035 !important;
    border-radius: 10px !important;
    color: #fff !important;
}
div[data-testid="stNumberInput"] label,
div[data-testid="stTextInput"] label {
    color: #4a5568 !important;
    font-size: 12px !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}
.stButton > button {
    background: linear-gradient(135deg, #1a5fa8, #378ADD) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.9 !important; }

.stDataFrame { background: #0d1117 !important; border-radius: 12px !important; }

[data-testid="stFileUploader"] {
    background: #0d1117 !important;
    border: 1px dashed #1a2035 !important;
    border-radius: 12px !important;
    padding: 20px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Cached functions ──────────────────────────────
@st.cache_data
def load_default_data():
    return pd.read_excel("personal_transactions_dashboard_ready (2).xlsx")

@st.cache_data
def load_uploaded_data(file_bytes):
    import io
    return pd.read_excel(io.BytesIO(file_bytes))

@st.cache_data
def prepare_monthly(_df):
    df = _df.copy()
    df['Month'] = pd.to_datetime(df['Month'])
    monthly = df.groupby('Month').apply(lambda g: pd.Series({
        'income'  : g[g['Transaction Type'] == 'credit']['Amount'].sum(),
        'expenses': g[g['Transaction Type'] == 'debit' ]['Amount'].sum(),
    })).reset_index()
    monthly['savings'] = monthly['income'] - monthly['expenses']
    monthly['debt']    = monthly['expenses'] * 0.10

    def score(income, expenses, savings, debt):
        s = 0
        er = expenses / income if income > 0 else 1
        sr = savings  / income if income > 0 else 0
        di = debt     / income if income > 0 else 1
        if   er <= 0.50: s += 40
        elif er <= 0.70: s += 30
        elif er <= 0.90: s += 15
        if   sr >= 0.30: s += 35
        elif sr >= 0.20: s += 28
        elif sr >= 0.10: s += 15
        elif sr >= 0.00: s +=  5
        if   di <= 0.15: s += 25
        elif di <= 0.36: s += 18
        elif di <= 0.50: s +=  8
        return min(100, s)

    monthly['health_score'] = monthly.apply(
        lambda r: score(r['income'], r['expenses'], r['savings'], r['debt']), axis=1)
    return monthly

@st.cache_resource
def train_models(X_tuple, y_tuple):
    X = np.array(X_tuple)
    y = np.array(y_tuple)
    lr  = LinearRegression()
    lr.fit(X, y)
    xgb = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    xgb.fit(X, y)
    return lr, xgb

# ── Hero ──────────────────────────────────────────
st.markdown("""
<div class="hero">
    <p class="hero-title">FinAI</p>
    <p class="hero-sub">AI-powered financial health & goal planning</p>
</div>
""", unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────
st.markdown('<p class="sec">Transaction Data</p>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload your bank Excel file (.xlsx)", type=["xlsx"], label_visibility="collapsed")

if uploaded_file:
    df = load_uploaded_data(uploaded_file.read())
    st.success(f"✅ {df.shape[0]} transactions loaded successfully")
else:
    df = load_default_data()
    st.caption("Using sample data — upload your own Excel file for personalised results")

monthly  = prepare_monthly(df)
X        = monthly[['income','expenses','savings','debt']].values
y        = monthly['health_score'].values
lr_model, xgb_model = train_models(tuple(map(tuple, X)), tuple(y))

cat_spending  = df[df['Transaction Type'] == 'debit'].groupby('Category')['Amount'].sum().sort_values(ascending=False)
top_3_cats    = list(cat_spending.head(3).index)
top_3_amounts = list(cat_spending.head(3).values)
best_months   = monthly[monthly['health_score'] >= 75]
avg_sav_best  = best_months['savings'].mean()  if len(best_months) > 0 else 0
avg_exp_best  = best_months['expenses'].mean() if len(best_months) > 0 else 0

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── Inputs ────────────────────────────────────────
st.markdown('<p class="sec">Your Finances</p>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
with c1: income   = st.number_input("Monthly Income (₹)",   min_value=0.0, value=10000.0, step=500.0)
with c2: expenses = st.number_input("Monthly Expenses (₹)", min_value=0.0, value=5000.0,  step=500.0)
with c3: savings  = st.number_input("Current Savings (₹)",  min_value=0.0, value=4000.0,  step=500.0)
with c4: debt     = st.number_input("Total Debt (₹)",       min_value=0.0, value=1000.0,  step=500.0)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── Goals ─────────────────────────────────────────
st.markdown('<p class="sec">Your Goals</p>', unsafe_allow_html=True)
num_goals = st.radio("", [1, 2], horizontal=True,
                     format_func=lambda x: f"{'One Goal' if x==1 else 'Two Goals'}")

goals = []
cols  = st.columns(num_goals)
for i in range(num_goals):
    with cols[i]:
        st.markdown(f"""<div class="goal-card">
            <p style="color:#378ADD;font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:1px;margin:0 0 16px;">Goal {i+1}</p>
        """, unsafe_allow_html=True)
        n = st.text_input("What is your goal?",
                          value="Buy a car" if i==0 else "Trip to Europe", key=f"gn{i}")
        c = st.number_input("How much does it cost? (₹)",
                            min_value=0.0, value=1500000.0 if i==0 else 200000.0,
                            step=10000.0, key=f"gc{i}")
        m = st.number_input("In how many months?",
                            min_value=1.0, value=24.0 if i==0 else 12.0,
                            step=1.0, key=f"gm{i}")
        goals.append({"name": n, "cost": c, "months": m})
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
go = st.button("Analyse My Finances", use_container_width=True)

if go:
    cf          = income - expenses
    lr_score    = float(np.clip(lr_model.predict([[income, expenses, savings, debt]])[0],  0, 100))
    xgb_score   = float(np.clip(xgb_model.predict([[income, expenses, savings, debt]])[0], 0, 100))
    final_score = (lr_score + xgb_score) / 2

    def get_label(s):
        if   s >= 75: return "Excellent", "#2ecc71"
        elif s >= 50: return "Good",      "#378ADD"
        elif s >= 30: return "Fair",      "#f39c12"
        else:         return "Poor",      "#e74c3c"

    label, hex_color = get_label(final_score)
    er  = expenses / income * 100
    sr  = savings  / income * 100
    dti = debt     / income * 100

    ideal_lr    = float(np.clip(lr_model.predict([[income, income*0.5, income*0.2, debt]])[0],  0, 100))
    ideal_xgb   = float(np.clip(xgb_model.predict([[income, income*0.5, income*0.2, debt]])[0], 0, 100))
    ideal_score = (ideal_lr + ideal_xgb) / 2
    exp_gap     = expenses - income * 0.50
    sav_gap     = income * 0.20 - savings

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Score ──────────────────────────────────────
    st.markdown('<p class="sec">Financial Health Score</p>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(f"""
        <div class="score-ring-wrap">
            <div style="font-size:11px;color:#4a5568;text-transform:uppercase;letter-spacing:2px;margin-bottom:16px;">Health Score</div>
            <div class="score-big" style="color:{hex_color}">{final_score:.1f}</div>
            <div class="score-sub">out of 100</div>
            <div class="score-lbl" style="color:{hex_color}">{label}</div>
            <div style="margin-top:20px;background:#131720;border-radius:999px;height:6px;overflow:hidden;">
                <div style="background:{hex_color};width:{final_score}%;height:100%;border-radius:999px;"></div>
            </div>
        </div>""", unsafe_allow_html=True)

    with c2:
        if   final_score >= 75: st.success("Your finances are in excellent shape!")
        elif final_score >= 50: st.info("Good shape — room to improve.")
        elif final_score >= 30: st.warning("Needs some attention.")
        else:                   st.error("Needs immediate attention.")

        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        metrics = [
            ("💸", f"{er:.1f}%",  "Expense Ratio",  er<=70,  "≤70% ideal"),
            ("🏦", f"{sr:.1f}%",  "Savings Rate",   sr>=20,  "≥20% ideal"),
            ("💳", f"{dti:.1f}%", "Debt/Income",    dti<=36, "≤36% ideal"),
            ("💵", f"₹{cf:,.0f}", "Monthly Surplus", cf>=0,  "income-expenses"),
        ]
        for col, (icon, val, lbl, good, hint) in zip([m1,m2,m3,m4], metrics):
            color = "#2ecc71" if good else "#e74c3c"
            with col:
                st.markdown(f"""<div class="card">
                    <div class="card-icon">{icon}</div>
                    <div class="card-val" style="color:{color}">{val}</div>
                    <div class="card-lbl">{lbl}</div>
                    <div style="font-size:10px;color:#2a3545;margin-top:4px;">{hint}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Goals ──────────────────────────────────────
    st.markdown('<p class="sec">Goal Tracker</p>', unsafe_allow_html=True)

    goal_results = []
    total_needed = 0

    for i, goal in enumerate(goals):
        remaining        = max(0, goal['cost'] - savings)
        needed_per_month = remaining / goal['months'] if goal['months'] > 0 else remaining
        total_needed    += needed_per_month
        achievable       = cf >= needed_per_month
        realistic_months = remaining / cf if cf > 0 else 9999
        completion_date  = datetime.now() + relativedelta(months=int(realistic_months))
        target_date      = datetime.now() + relativedelta(months=int(goal['months']))
        progress_pct     = min(100, savings / goal['cost'] * 100) if goal['cost'] > 0 else 0
        bar_color        = "#2ecc71" if progress_pct >= 50 else "#f39c12" if progress_pct >= 25 else "#378ADD"

        goal_results.append({
            "goal": goal, "remaining": remaining,
            "needed_per_month": needed_per_month, "achievable": achievable,
            "realistic_months": realistic_months, "completion_date": completion_date,
            "progress_pct": progress_pct
        })

        st.markdown(f"""
        <div class="goal-card">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">
                <div>
                    <p style="color:#fff;font-size:18px;font-weight:700;margin:0;">🎯 {goal['name']}</p>
                    <p style="color:#4a5568;font-size:12px;margin:4px 0 0;">Target: ₹{goal['cost']:,.0f} in {goal['months']:.0f} months</p>
                </div>
                <div style="text-align:right;">
                    <p style="color:{bar_color};font-size:22px;font-weight:800;margin:0;">{progress_pct:.1f}%</p>
                    <p style="color:#4a5568;font-size:11px;margin:0;">complete</p>
                </div>
            </div>
            <div class="progress-track">
                <div class="progress-fill" style="width:{progress_pct}%;background:{bar_color};"></div>
            </div>
            <div style="display:flex;justify-content:space-between;margin-top:6px;">
                <span style="color:#4a5568;font-size:11px;">Saved: ₹{savings:,.0f}</span>
                <span style="color:#4a5568;font-size:11px;">Remaining: ₹{remaining:,.0f}</span>
            </div>
        </div>""", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class="card">
                <div class="card-icon">📅</div>
                <div class="card-val">₹{needed_per_month:,.0f}</div>
                <div class="card-lbl">Needed / Month</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="card">
                <div class="card-icon">💰</div>
                <div class="card-val" style="color:{'#2ecc71' if achievable else '#e74c3c'}">₹{cf:,.0f}</div>
                <div class="card-lbl">Your Surplus</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="card">
                <div class="card-icon">🏁</div>
                <div class="card-val">{completion_date.strftime('%b %Y')}</div>
                <div class="card-lbl">Est. Completion</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            months_away = int(realistic_months)
            st.markdown(f"""<div class="card">
                <div class="card-icon">⏳</div>
                <div class="card-val" style="color:{'#2ecc71' if achievable else '#f39c12'}">{months_away}</div>
                <div class="card-lbl">Months Away</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if achievable:
            st.success(f"✅ You can achieve this by {target_date.strftime('%B %Y')} with your current surplus!")
        else:
            shortfall = needed_per_month - cf
            st.error(f"❌ Not achievable in {goal['months']:.0f} months — realistic timeline is {realistic_months:.0f} months")
            st.info(f"💡 Reduce expenses by ₹{shortfall:,.0f}/month to hit your original target")

        st.markdown(f"**🔧 Savings Simulator for {goal['name']}**")
        extra = st.slider(f"If I save ₹___ more per month",
                          0, int(income), 0, 500, key=f"sl{i}",
                          format="₹%d")
        if extra > 0:
            new_months   = remaining / (cf + extra) if (cf + extra) > 0 else 9999
            new_date     = datetime.now() + relativedelta(months=int(new_months))
            saved_months = max(0, realistic_months - new_months)
            st.success(f"💡 Saving ₹{extra:,.0f} more → Goal achieved by **{new_date.strftime('%B %Y')}** — {saved_months:.0f} months earlier!")

        st.markdown(f"**📅 Month by Month Savings Plan**")
        plan      = []
        running   = savings
        for m in range(1, min(int(goal['months']) + 3, 13)):
            month_save = min(needed_per_month, cf)
            running   += month_save
            pct        = min(100, running / goal['cost'] * 100)
            plan.append({
                'Month'           : f'Month {m}',
                'Save This Month' : f'₹{month_save:,.0f}',
                'Total Saved'     : f'₹{running:,.0f}',
                'Progress'        : f'{pct:.1f}%'
            })
            if running >= goal['cost']:
                break
        st.dataframe(pd.DataFrame(plan), use_container_width=True, hide_index=True)
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Combined
    if len(goals) == 2:
        st.markdown('<p class="sec">Combined Goal Summary</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class="card">
                <div class="card-icon">🎯</div>
                <div class="card-val">₹{sum(g['cost'] for g in goals):,.0f}</div>
                <div class="card-lbl">Total Goal Amount</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="card">
                <div class="card-icon">📉</div>
                <div class="card-val">₹{sum(r['remaining'] for r in goal_results):,.0f}</div>
                <div class="card-lbl">Total Remaining</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="card">
                <div class="card-icon">💸</div>
                <div class="card-val" style="color:{'#2ecc71' if cf>=total_needed else '#e74c3c'}">₹{total_needed:,.0f}</div>
                <div class="card-lbl">Total Needed/Month</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if cf >= total_needed:
            st.success(f"✅ Your monthly surplus of ₹{cf:,.0f} is enough to cover both goals!")
        else:
            st.error(f"❌ You need ₹{total_needed - cf:,.0f} more per month to achieve both goals together")
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Budget Rule ────────────────────────────────
    st.markdown('<p class="sec">50 / 30 / 20 Budget Rule</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        needs_pct  = min(100, expenses / (income*0.5) * 100)  if income > 0 else 0
        wants_pct  = 100
        savings_pct= min(100, savings  / (income*0.2) * 100)  if income > 0 else 0

        st.markdown(f"""
        <div class="goal-card">
            <p style="color:#4a5568;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin:0 0 16px;">
                Recommended split of ₹{income:,.0f}/month
            </p>
            <div class="budget-row">
                <div>
                    <p style="color:#fff;font-size:14px;font-weight:600;margin:0;">Needs</p>
                    <p style="color:#4a5568;font-size:11px;margin:2px 0 0;">50% of income</p>
                </div>
                <div style="text-align:right;">
                    <p style="color:#378ADD;font-size:16px;font-weight:700;margin:0;">₹{income*0.5:,.0f}</p>
                    <p style="color:{'#2ecc71' if expenses<=income*0.5 else '#e74c3c'};font-size:11px;margin:2px 0 0;">
                        You spend ₹{expenses:,.0f}
                    </p>
                </div>
            </div>
            <div class="budget-row">
                <div>
                    <p style="color:#fff;font-size:14px;font-weight:600;margin:0;">Wants</p>
                    <p style="color:#4a5568;font-size:11px;margin:2px 0 0;">30% of income</p>
                </div>
                <div style="text-align:right;">
                    <p style="color:#BA7517;font-size:16px;font-weight:700;margin:0;">₹{income*0.3:,.0f}</p>
                </div>
            </div>
            <div class="budget-row">
                <div>
                    <p style="color:#fff;font-size:14px;font-weight:600;margin:0;">Savings</p>
                    <p style="color:#4a5568;font-size:11px;margin:2px 0 0;">20% of income</p>
                </div>
                <div style="text-align:right;">
                    <p style="color:#2ecc71;font-size:16px;font-weight:700;margin:0;">₹{income*0.2:,.0f}</p>
                    <p style="color:{'#2ecc71' if savings>=income*0.2 else '#e74c3c'};font-size:11px;margin:2px 0 0;">
                        You save ₹{savings:,.0f}
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if exp_gap > 0: st.error(f"❌ Reduce expenses by ₹{exp_gap:,.0f}")
        else:           st.success(f"✅ Expenses ₹{abs(exp_gap):,.0f} under the 50% limit")
        if sav_gap > 0: st.error(f"❌ Save ₹{sav_gap:,.0f} more to hit 20% target")
        else:           st.success(f"✅ Savings ₹{abs(sav_gap):,.0f} above 20% target")
        if ideal_score > final_score:
            st.info(f"💡 Following 50/30/20 improves your score by {ideal_score-final_score:.1f} pts")

    with c2:
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='#0d1117')
        ax.set_facecolor('#0d1117')
        cats = ['Needs\n50%','Wants\n30%','Savings\n20%','Your\nExpenses','Your\nSavings']
        vals = [income*0.5, income*0.3, income*0.2, expenses, savings]
        clrs = ['#1a5fa8','#7a4f00','#1a6b3a','#8b1a1a','#1a6b3a']
        bars = ax.bar(cats, vals, color=clrs, edgecolor='#0d1117', width=0.55)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+50,
                    f'₹{val:,.0f}', ha='center', color='#888', fontsize=8)
        ax.tick_params(colors='#333', labelsize=9)
        for s in ['top','right','left']: ax.spines[s].set_visible(False)
        ax.spines['bottom'].set_color('#1a2035')
        ax.yaxis.set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── 6 Month Projection ─────────────────────────
    st.markdown('<p class="sec">6 Month Projection</p>', unsafe_allow_html=True)
    pe_vals = [expenses*(1-0.01*i) for i in range(1,7)]
    ps_vals = [savings *(1+0.01*i) for i in range(1,7)]
    ms      = [f'M{i}' for i in range(1,7)]
    c1, c2  = st.columns(2)
    with c1:
        rows = [{'Month': ms[i],
                 'Expenses': f'₹{pe_vals[i]:,.0f}',
                 'Savings' : f'₹{ps_vals[i]:,.0f}',
                 'Surplus' : f'₹{income-pe_vals[i]:,.0f}'} for i in range(6)]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    with c2:
        fig, ax = plt.subplots(figsize=(5, 3.5), facecolor='#0d1117')
        ax.set_facecolor('#0d1117')
        ax.plot(ms, pe_vals, marker='o', color='#e74c3c', linewidth=2.5,
                markersize=6, label='Expenses')
        ax.plot(ms, ps_vals, marker='s', color='#2ecc71', linewidth=2.5,
                markersize=6, label='Savings')
        ax.fill_between(ms, pe_vals, alpha=0.05, color='#e74c3c')
        ax.fill_between(ms, ps_vals, alpha=0.05, color='#2ecc71')
        ax.tick_params(colors='#333', labelsize=9)
        for s in ['top','right']: ax.spines[s].set_visible(False)
        ax.spines['bottom'].set_color('#1a2035')
        ax.spines['left'].set_color('#1a2035')
        ax.legend(facecolor='#0d1117', labelcolor='#888', fontsize=9, framealpha=0.5)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'₹{x:,.0f}'))
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Category ───────────────────────────────────
    st.markdown('<p class="sec">Where Is Your Money Going?</p>', unsafe_allow_html=True)
    cat_df = df[df['Transaction Type']=='debit'].groupby('Category')['Amount'].sum().sort_values(ascending=False).head(8)
    c1, c2 = st.columns(2)
    with c1:
        cd = cat_df.reset_index()
        cd.columns = ['Category','Spent']
        cd['Spent'] = cd['Spent'].apply(lambda x: f'₹{x:,.0f}')
        st.dataframe(cd, use_container_width=True, hide_index=True)
    with c2:
        fig, ax = plt.subplots(figsize=(5, 3.5), facecolor='#0d1117')
        ax.set_facecolor('#0d1117')
        bar_colors = plt.cm.Blues(np.linspace(0.35, 0.85, len(cat_df)))
        cat_df.plot(kind='barh', ax=ax, color=bar_colors, edgecolor='#0d1117')
        ax.tick_params(colors='#333', labelsize=9)
        for s in ['top','right']: ax.spines[s].set_visible(False)
        ax.spines['bottom'].set_color('#1a2035')
        ax.spines['left'].set_color('#1a2035')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'₹{x:,.0f}'))
        ax.set_xlabel(''); ax.set_ylabel('')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── AI Advisor ─────────────────────────────────
    st.markdown('<p class="sec">AI Financial Advisor</p>', unsafe_allow_html=True)
    with st.spinner("Analysing your finances and goals..."):
        try:
            goals_text = "\n".join([
                f"- {g['goal']['name']}: costs Rs {g['goal']['cost']:,.0f}, timeline {g['goal']['months']:.0f} months, needs Rs {g['needed_per_month']:,.0f}/month, {'achievable' if g['achievable'] else 'NOT achievable in time'}"
                for g in goal_results
            ])
            prompt = f"""
You are a friendly personal financial advisor. Talk like a helpful friend, not a robot.

USER FINANCES:
- Monthly Income: Rs {income:,.0f}
- Monthly Expenses: Rs {expenses:,.0f}
- Current Savings: Rs {savings:,.0f}
- Total Debt: Rs {debt:,.0f}
- Monthly Surplus: Rs {cf:,.0f}
- Financial Health Score: {final_score:.1f}/100 ({label})

USER GOALS:
{goals_text}

TOP 3 SPENDING CATEGORIES:
- {top_3_cats[0]}: Rs {top_3_amounts[0]:,.0f}
- {top_3_cats[1]}: Rs {top_3_amounts[1]:,.0f}
- {top_3_cats[2]}: Rs {top_3_amounts[2]:,.0f}

Give exactly 5 tips. Each tip should:
1. Be specific to their numbers
2. Mention exact rupee amounts
3. Help them reach their goals faster
4. Be written in simple friendly language
No jargon. No mention of ML, models, algorithms. Just practical money advice.
"""
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}]
            )
            tips_text = response.choices[0].message.content
            st.markdown(f"""
            <div class="goal-card">
                <div style="color:#4a5568;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin-bottom:16px;">
                    Personalised advice based on your data
                </div>
                {tips_text}
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Could not load AI recommendations: {e}")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── PDF ────────────────────────────────────────
    st.markdown('<p class="sec">Download Report</p>', unsafe_allow_html=True)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 14, "FinAI Financial Health Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Generated on {datetime.now().strftime('%d %B %Y')}", ln=True, align="C")
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 9, "Your Finances", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, f"Monthly Income   : Rs {income:,.0f}", ln=True)
    pdf.cell(0, 7, f"Monthly Expenses : Rs {expenses:,.0f}", ln=True)
    pdf.cell(0, 7, f"Current Savings  : Rs {savings:,.0f}", ln=True)
    pdf.cell(0, 7, f"Total Debt       : Rs {debt:,.0f}", ln=True)
    pdf.cell(0, 7, f"Monthly Surplus  : Rs {cf:,.0f}", ln=True)
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 9, f"Financial Health Score: {final_score:.1f} / 100 ({label})", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, f"Expense Ratio: {er:.1f}%  |  Savings Rate: {sr:.1f}%  |  Debt-to-Income: {dti:.1f}%", ln=True)
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 9, "Your Goals", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for i, gr in enumerate(goal_results):
        pdf.cell(0, 7, f"Goal {i+1}: {gr['goal']['name']}", ln=True)
        pdf.cell(0, 7, f"  Cost: Rs {gr['goal']['cost']:,.0f}", ln=True)
        pdf.cell(0, 7, f"  Remaining: Rs {gr['remaining']:,.0f}", ln=True)
        pdf.cell(0, 7, f"  Needed per month: Rs {gr['needed_per_month']:,.0f}", ln=True)
        pdf.cell(0, 7, f"  Status: {'Achievable' if gr['achievable'] else 'Needs more savings'}", ln=True)
        pdf.cell(0, 7, f"  Estimated completion: {gr['completion_date'].strftime('%B %Y')}", ln=True)
        pdf.ln(2)

    pdf_bytes = pdf.output()
    st.download_button(
        "Download My Report",
        bytes(pdf_bytes),
        "finai_report.pdf",
        "application/pdf",
        use_container_width=True
    )
