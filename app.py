import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from groq import Groq
from datetime import datetime
from dateutil.relativedelta import relativedelta
import streamlit.components.v1 as components
import warnings
warnings.filterwarnings('ignore')

client = Groq(api_key="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")

st.set_page_config(page_title="FinAI", page_icon="💰", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
* { font-family: 'Inter', sans-serif !important; margin: 0; padding: 0; box-sizing: border-box; }
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: #050810 !important;
    color: #fff;
}
[data-testid="stHeader"] { display: none; }
[data-testid="stSidebar"] { display: none; }
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stMain"] > div { padding: 0 !important; }
div[data-testid="stNumberInput"] > div > div > input,
div[data-testid="stTextInput"] > div > div > input {
    background: #0d1117 !important;
    border: 1px solid #1e2535 !important;
    border-radius: 12px !important;
    color: #fff !important;
    font-size: 15px !important;
    padding: 14px 16px !important;
}
div[data-testid="stNumberInput"] > div > div > input:focus,
div[data-testid="stTextInput"] > div > div > input:focus {
    border-color: #378ADD !important;
    box-shadow: 0 0 0 3px rgba(55,138,221,0.15) !important;
}
div[data-testid="stNumberInput"] label,
div[data-testid="stTextInput"] label,
div[data-testid="stSelectbox"] label {
    color: #4a5568 !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
    font-weight: 600 !important;
}
.stButton > button {
    background: linear-gradient(135deg, #1a5fa8 0%, #378ADD 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 16px 40px !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 8px 32px rgba(55,138,221,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 40px rgba(55,138,221,0.4) !important;
}
[data-testid="stRadio"] label { color: #fff !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────
if 'screen' not in st.session_state:        st.session_state.screen = 1
if 'user_data' not in st.session_state:     st.session_state.user_data = {}
if 'finance_data' not in st.session_state:  st.session_state.finance_data = {}
if 'active_bubble' not in st.session_state: st.session_state.active_bubble = None

# ── Cached functions ──────────────────────────────
@st.cache_data
def load_default_data():
    return pd.read_excel("personal_transactions_dashboard_ready (2).xlsx")

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
        er = expenses/income if income>0 else 1
        sr = savings/income  if income>0 else 0
        di = debt/income     if income>0 else 1
        if er<=0.50: s+=40
        elif er<=0.70: s+=30
        elif er<=0.90: s+=15
        if sr>=0.30: s+=35
        elif sr>=0.20: s+=28
        elif sr>=0.10: s+=15
        elif sr>=0.00: s+=5
        if di<=0.15: s+=25
        elif di<=0.36: s+=18
        elif di<=0.50: s+=8
        return min(100, s)
    monthly['health_score'] = monthly.apply(
        lambda r: score(r['income'],r['expenses'],r['savings'],r['debt']), axis=1)
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

df       = load_default_data()
monthly  = prepare_monthly(df)
X        = monthly[['income','expenses','savings','debt']].values
y        = monthly['health_score'].values
lr_model, xgb_model = train_models(tuple(map(tuple, X)), tuple(y))
cat_spending  = df[df['Transaction Type']=='debit'].groupby('Category')['Amount'].sum().sort_values(ascending=False)
top_3_cats    = list(cat_spending.head(3).index)
top_3_amounts = list(cat_spending.head(3).values)
best_months   = monthly[monthly['health_score']>=75]
avg_sav_best  = best_months['savings'].mean()  if len(best_months)>0 else 0
avg_exp_best  = best_months['expenses'].mean() if len(best_months)>0 else 0

# ════════════════════════════════════════════════
# SCREEN 1 — WELCOME
# ════════════════════════════════════════════════
if st.session_state.screen == 1:
    components.html("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800;900&display=swap');
    * { margin:0; padding:0; box-sizing:border-box; font-family:'Inter',sans-serif; }
    body { background: #050810; overflow: hidden; }

    .bg {
        position: fixed; inset: 0;
        background: radial-gradient(ellipse at 30% 20%, rgba(55,138,221,0.12) 0%, transparent 60%),
                    radial-gradient(ellipse at 80% 80%, rgba(100,60,200,0.08) 0%, transparent 50%);
    }

    .orb {
        position: fixed;
        border-radius: 50%;
        filter: blur(80px);
        animation: float 8s ease-in-out infinite;
    }
    .orb1 { width:400px; height:400px; background:rgba(55,138,221,0.08); top:-100px; right:-100px; animation-delay:0s; }
    .orb2 { width:300px; height:300px; background:rgba(100,60,200,0.06); bottom:-50px; left:-50px; animation-delay:3s; }
    .orb3 { width:200px; height:200px; background:rgba(55,221,138,0.05); top:50%; left:50%; animation-delay:5s; }

    @keyframes float {
        0%,100% { transform: translateY(0px) scale(1); }
        50% { transform: translateY(-30px) scale(1.05); }
    }

    .wrap {
        position: relative; z-index: 10;
        min-height: 100vh;
        display: flex; flex-direction: column;
        align-items: center; justify-content: center;
        padding: 40px 20px;
    }

    .badge {
        display: inline-flex; align-items: center; gap: 8px;
        background: rgba(55,138,221,0.1);
        border: 1px solid rgba(55,138,221,0.2);
        border-radius: 999px;
        padding: 8px 20px;
        font-size: 12px; font-weight: 600;
        color: #378ADD; letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 32px;
        animation: fadeUp 0.8s ease forwards;
    }
    .dot { width:6px; height:6px; background:#378ADD; border-radius:50%; animation: pulse 2s infinite; }
    @keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.5;transform:scale(1.5)} }

    .title {
        font-size: clamp(48px, 8vw, 88px);
        font-weight: 900;
        line-height: 1;
        text-align: center;
        margin-bottom: 20px;
        animation: fadeUp 0.8s ease 0.2s both;
    }
    .title span {
        background: linear-gradient(135deg, #fff 0%, #378ADD 50%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .sub {
        font-size: 18px; color: #4a5568;
        text-align: center; max-width: 480px;
        line-height: 1.6; margin-bottom: 60px;
        animation: fadeUp 0.8s ease 0.4s both;
    }

    .stats {
        display: flex; gap: 40px;
        margin-bottom: 60px;
        animation: fadeUp 0.8s ease 0.6s both;
    }
    .stat { text-align: center; }
    .stat-num { font-size: 28px; font-weight: 800; color: #fff; }
    .stat-lbl { font-size: 11px; color: #4a5568; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }

    .scroll {
        display: flex; flex-direction: column; align-items: center; gap: 8px;
        color: #2a3545; font-size: 12px; letter-spacing: 2px;
        text-transform: uppercase;
        animation: fadeUp 0.8s ease 0.8s both;
    }
    .arrow { animation: bounce 2s infinite; }
    @keyframes bounce { 0%,100%{transform:translateY(0)} 50%{transform:translateY(8px)} }

    @keyframes fadeUp {
        from { opacity:0; transform:translateY(30px); }
        to   { opacity:1; transform:translateY(0); }
    }

    .grid {
        position: fixed; inset: 0;
        background-image: linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px),
                          linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px);
        background-size: 60px 60px;
        pointer-events: none;
    }
    </style>

    <div class="bg"></div>
    <div class="grid"></div>
    <div class="orb orb1"></div>
    <div class="orb orb2"></div>
    <div class="orb orb3"></div>

    <div class="wrap">
        <div class="badge"><div class="dot"></div>AI POWERED FINANCE</div>
        <h1 class="title"><span>Know Your<br>Financial Health</span></h1>
        <p class="sub">Answer a few questions and get a personalised financial health score with an AI-powered plan to achieve your goals.</p>
        <div class="stats">
            <div class="stat"><div class="stat-num">806</div><div class="stat-lbl">Transactions Analysed</div></div>
            <div class="stat"><div class="stat-num">21</div><div class="stat-lbl">Months of Data</div></div>
            <div class="stat"><div class="stat-num">2</div><div class="stat-lbl">ML Models</div></div>
        </div>
        <div class="scroll">
            <span>scroll to begin</span>
            <div class="arrow">↓</div>
        </div>
    </div>
    """, height=600)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("""
        <div style='text-align:center; margin-bottom:32px;'>
            <p style='color:#4a5568; font-size:11px; text-transform:uppercase; letter-spacing:2px; font-weight:600;'>
                Let's get started
            </p>
            <h2 style='color:#fff; font-size:28px; font-weight:800; margin-top:8px;'>Tell us about yourself</h2>
        </div>
        """, unsafe_allow_html=True)

        name = st.text_input("Your Name", placeholder="e.g. Kashish Gupta")
        age  = st.number_input("Your Age", min_value=18, max_value=80, value=22, step=1)
        occ  = st.selectbox("Occupation", [
            "Student", "Salaried Employee", "Self Employed",
            "Freelancer", "Business Owner", "Other"
        ])

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Continue →"):
            if name.strip() == "":
                st.error("Please enter your name to continue")
            else:
                st.session_state.user_data = {"name": name, "age": age, "occupation": occ}
                st.session_state.screen = 2
                st.rerun()

# ════════════════════════════════════════════════
# SCREEN 2 — FINANCIAL DATA + GOALS
# ════════════════════════════════════════════════
elif st.session_state.screen == 2:
    name = st.session_state.user_data.get("name", "there")

    components.html(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800;900&display=swap');
    * {{ margin:0; padding:0; box-sizing:border-box; font-family:'Inter',sans-serif; }}
    body {{ background:#050810; }}
    .bg {{
        position:fixed; inset:0;
        background: radial-gradient(ellipse at 70% 30%, rgba(55,138,221,0.1) 0%, transparent 60%);
    }}
    .grid {{
        position:fixed; inset:0;
        background-image: linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px),
                          linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px);
        background-size: 60px 60px;
        pointer-events:none;
    }}
    .header {{
        position:relative; z-index:10;
        padding: 40px 40px 20px;
        animation: fadeUp 0.6s ease forwards;
    }}
    .greeting {{
        font-size: 13px; color: #378ADD;
        text-transform: uppercase; letter-spacing: 2px;
        font-weight: 600; margin-bottom: 8px;
    }}
    .title {{
        font-size: 36px; font-weight: 900; color: #fff;
    }}
    .step {{
        display:flex; gap:8px; margin-top:16px;
    }}
    .step-dot {{
        width:8px; height:8px; border-radius:50%;
        background:#1e2535;
    }}
    .step-dot.done {{ background:#378ADD; }}
    .step-dot.active {{
        background:#378ADD;
        box-shadow:0 0 8px rgba(55,138,221,0.6);
    }}
    @keyframes fadeUp {{
        from {{ opacity:0; transform:translateY(20px); }}
        to   {{ opacity:1; transform:translateY(0); }}
    }}
    </style>
    <div class="bg"></div>
    <div class="grid"></div>
    <div class="header">
        <div class="greeting">Welcome, {name}</div>
        <div class="title">Your Financial Details</div>
        <div class="step">
            <div class="step-dot done"></div>
            <div class="step-dot active"></div>
            <div class="step-dot"></div>
        </div>
    </div>
    """, height=160)

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown("""
        <div style='background:#0d1117; border-radius:20px; padding:32px; border:1px solid #1e2535; margin-bottom:24px;'>
            <p style='color:#378ADD; font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:2px; margin-bottom:20px;'>
                Monthly Finances
            </p>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            income   = st.number_input("Monthly Income (₹)",   min_value=0.0, value=10000.0, step=500.0)
            savings  = st.number_input("Current Savings (₹)",  min_value=0.0, value=4000.0,  step=500.0)
        with c2:
            expenses = st.number_input("Monthly Expenses (₹)", min_value=0.0, value=5000.0,  step=500.0)
            debt     = st.number_input("Total Debt (₹)",       min_value=0.0, value=1000.0,  step=500.0)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div style='background:#0d1117; border-radius:20px; padding:32px; border:1px solid #1e2535; margin-bottom:24px;'>
            <p style='color:#a78bfa; font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:2px; margin-bottom:20px;'>
                Your Goals
            </p>
        </div>
        """, unsafe_allow_html=True)

        num_goals = st.radio("", ["1 Goal", "2 Goals"], horizontal=True)
        n_goals   = 1 if num_goals == "1 Goal" else 2

        goals = []
        gcols = st.columns(n_goals)
        for i in range(n_goals):
            with gcols[i]:
                st.markdown(f"<p style='color:#a78bfa;font-size:12px;font-weight:600;margin-bottom:8px;'>Goal {i+1}</p>", unsafe_allow_html=True)
                n = st.text_input("Goal name",     value="Buy a car" if i==0 else "Europe trip", key=f"gn{i}")
                c = st.number_input("Cost (₹)",    min_value=0.0, value=1500000.0 if i==0 else 200000.0, step=10000.0, key=f"gc{i}")
                m = st.number_input("In months",   min_value=1.0, value=24.0 if i==0 else 12.0, step=1.0, key=f"gm{i}")
                goals.append({"name": n, "cost": c, "months": m})

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Analyse My Finances ✦"):
            st.session_state.finance_data = {
                "income": income, "expenses": expenses,
                "savings": savings, "debt": debt, "goals": goals
            }
            st.session_state.screen = 3
            st.session_state.active_bubble = None
            st.rerun()

# ════════════════════════════════════════════════
# SCREEN 3 — DASHBOARD WITH BUBBLES
# ════════════════════════════════════════════════
elif st.session_state.screen == 3:
    fd   = st.session_state.finance_data
    ud   = st.session_state.user_data
    name = ud.get("name", "there")

    income   = fd['income']
    expenses = fd['expenses']
    savings  = fd['savings']
    debt     = fd['debt']
    goals    = fd['goals']
    cf       = income - expenses

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

    goal_results = []
    for goal in goals:
        remaining        = max(0, goal['cost'] - savings)
        needed_per_month = remaining / goal['months'] if goal['months'] > 0 else remaining
        achievable       = cf >= needed_per_month
        realistic_months = remaining / cf if cf > 0 else 9999
        completion_date  = datetime.now() + relativedelta(months=int(realistic_months))
        progress_pct     = min(100, savings / goal['cost'] * 100) if goal['cost'] > 0 else 0
        goal_results.append({
            "goal": goal, "remaining": remaining,
            "needed_per_month": needed_per_month, "achievable": achievable,
            "realistic_months": realistic_months, "completion_date": completion_date,
            "progress_pct": progress_pct
        })

    # Bubble dashboard
    bubble_html = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800;900&display=swap');
    * {{ margin:0; padding:0; box-sizing:border-box; font-family:'Inter',sans-serif; }}
    body {{ background:#050810; overflow-x:hidden; }}

    .bg {{
        position:fixed; inset:0;
        background: radial-gradient(ellipse at 50% 50%, rgba(55,138,221,0.08) 0%, transparent 70%);
    }}
    .grid {{
        position:fixed; inset:0;
        background-image: linear-gradient(rgba(255,255,255,0.015) 1px, transparent 1px),
                          linear-gradient(90deg, rgba(255,255,255,0.015) 1px, transparent 1px);
        background-size: 60px 60px; pointer-events:none;
    }}

    .header {{
        position:relative; z-index:10;
        display:flex; justify-content:space-between; align-items:center;
        padding:24px 40px;
        border-bottom:1px solid #0d1117;
    }}
    .logo {{ font-size:20px; font-weight:900; color:#fff; }}
    .logo span {{ color:#378ADD; }}
    .user-pill {{
        background:#0d1117; border:1px solid #1e2535;
        border-radius:999px; padding:8px 16px;
        font-size:13px; color:#888; font-weight:500;
    }}

    .scene {{
        position:relative;
        width:100%; height:600px;
        display:flex; align-items:center; justify-content:center;
    }}

    .center-score {{
        position:absolute;
        width:200px; height:200px;
        border-radius:50%;
        background: radial-gradient(circle, #0d1a2e 0%, #050810 100%);
        border:2px solid {hex_color}44;
        display:flex; flex-direction:column;
        align-items:center; justify-content:center;
        z-index:20;
        box-shadow: 0 0 60px {hex_color}22, inset 0 0 40px rgba(0,0,0,0.5);
        animation: scorePop 1s cubic-bezier(0.34,1.56,0.64,1) forwards;
    }}
    @keyframes scorePop {{
        from {{ transform:scale(0); opacity:0; }}
        to   {{ transform:scale(1); opacity:1; }}
    }}
    .score-ring {{
        position:absolute;
        width:220px; height:220px;
        border-radius:50%;
        border:1px solid {hex_color}33;
        animation: ringPulse 3s ease-in-out infinite;
    }}
    .score-ring2 {{
        position:absolute;
        width:260px; height:260px;
        border-radius:50%;
        border:1px solid {hex_color}15;
        animation: ringPulse 3s ease-in-out infinite 1s;
    }}
    @keyframes ringPulse {{
        0%,100% {{ transform:scale(1); opacity:1; }}
        50% {{ transform:scale(1.05); opacity:0.5; }}
    }}
    .score-num {{ font-size:52px; font-weight:900; color:{hex_color}; line-height:1; }}
    .score-of  {{ font-size:11px; color:#2a3545; margin-top:2px; }}
    .score-lbl {{ font-size:13px; font-weight:700; color:{hex_color}; margin-top:6px; }}

    .bubble {{
        position:absolute;
        border-radius:50%;
        display:flex; flex-direction:column;
        align-items:center; justify-content:center;
        cursor:pointer;
        transition:all 0.4s cubic-bezier(0.34,1.56,0.64,1);
        text-decoration:none;
        z-index:15;
    }}
    .bubble:hover {{
        transform:scale(1.12) !important;
        box-shadow:0 20px 60px rgba(0,0,0,0.5) !important;
    }}
    .bubble-icon {{ font-size:22px; margin-bottom:4px; }}
    .bubble-label {{ font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:1px; }}
    .bubble-sub {{ font-size:10px; margin-top:2px; opacity:0.7; }}

    .b1 {{
        width:120px; height:120px;
        background:linear-gradient(135deg,#0d2a4a,#1a4a7a);
        border:1px solid #378ADD44;
        color:#378ADD;
        top:80px; left:calc(50% - 240px);
        animation: floatBubble 1s cubic-bezier(0.34,1.56,0.64,1) 0.2s both;
    }}
    .b2 {{
        width:110px; height:110px;
        background:linear-gradient(135deg,#2a0d4a,#4a1a7a);
        border:1px solid #a78bfa44;
        color:#a78bfa;
        top:60px; right:calc(50% - 240px);
        animation: floatBubble 1s cubic-bezier(0.34,1.56,0.64,1) 0.3s both;
    }}
    .b3 {{
        width:130px; height:130px;
        background:linear-gradient(135deg,#0d2a1a,#1a4a2a);
        border:1px solid #2ecc7144;
        color:#2ecc71;
        bottom:80px; left:calc(50% - 260px);
        animation: floatBubble 1s cubic-bezier(0.34,1.56,0.64,1) 0.4s both;
    }}
    .b4 {{
        width:115px; height:115px;
        background:linear-gradient(135deg,#2a1a0d,#4a2a1a);
        border:1px solid #f39c1244;
        color:#f39c12;
        bottom:90px; right:calc(50% - 250px);
        animation: floatBubble 1s cubic-bezier(0.34,1.56,0.64,1) 0.5s both;
    }}
    .b5 {{
        width:105px; height:105px;
        background:linear-gradient(135deg,#2a0d1a,#4a1a2a);
        border:1px solid #e74c3c44;
        color:#e74c3c;
        top:50%; left:calc(50% - 320px);
        transform:translateY(-50%);
        animation: floatBubble 1s cubic-bezier(0.34,1.56,0.64,1) 0.6s both;
    }}
    .b6 {{
        width:115px; height:115px;
        background:linear-gradient(135deg,#0d1a2a,#1a2a3a);
        border:1px solid #00bcd444;
        color:#00bcd4;
        top:50%; right:calc(50% - 320px);
        transform:translateY(-50%);
        animation: floatBubble 1s cubic-bezier(0.34,1.56,0.64,1) 0.7s both;
    }}

    @keyframes floatBubble {{
        from {{ opacity:0; transform:scale(0) translateY(40px); }}
        to   {{ opacity:1; transform:scale(1) translateY(0); }}
    }}

    .b3, .b4 {{ animation-fill-mode:both; }}

    .float-anim {{
        animation: gentleFloat 4s ease-in-out infinite !important;
    }}
    .float-anim:nth-child(2) {{ animation-delay:0.5s !important; }}
    .float-anim:nth-child(3) {{ animation-delay:1s !important; }}
    .float-anim:nth-child(4) {{ animation-delay:1.5s !important; }}
    .float-anim:nth-child(5) {{ animation-delay:2s !important; }}
    .float-anim:nth-child(6) {{ animation-delay:2.5s !important; }}

    @keyframes gentleFloat {{
        0%,100% {{ transform:translateY(0); }}
        50% {{ transform:translateY(-10px); }}
    }}

    .hint {{
        position:absolute; bottom:20px;
        font-size:12px; color:#2a3545;
        letter-spacing:2px; text-transform:uppercase;
        animation: fadeIn 1s ease 1.5s both;
    }}
    @keyframes fadeIn {{
        from {{ opacity:0; }}
        to   {{ opacity:1; }}
    }}

    .connector {{
        position:absolute;
        background:linear-gradient(90deg,{hex_color}22,transparent);
        height:1px; z-index:5;
        transform-origin:left center;
        animation: growLine 0.8s ease forwards;
    }}
    @keyframes growLine {{
        from {{ transform:scaleX(0); }}
        to   {{ transform:scaleX(1); }}
    }}

    .name-tag {{
        position:absolute; top:20px; left:50%; transform:translateX(-50%);
        font-size:12px; color:#2a3545;
        text-transform:uppercase; letter-spacing:2px;
    }}
    </style>

    <div class="bg"></div>
    <div class="grid"></div>

    <div class="header">
        <div class="logo">Fin<span>AI</span></div>
        <div class="user-pill">👋 {name}</div>
    </div>

    <div class="scene">
        <div class="score-ring2"></div>
        <div class="score-ring"></div>

        <div class="center-score">
            <div class="score-num">{final_score:.0f}</div>
            <div class="score-of">out of 100</div>
            <div class="score-lbl">{label}</div>
        </div>

        <div class="bubble b1 float-anim" onclick="window.parent.postMessage('goals','*')">
            <div class="bubble-icon">🎯</div>
            <div class="bubble-label">Goals</div>
            <div class="bubble-sub">Tap to view</div>
        </div>

        <div class="bubble b2 float-anim" onclick="window.parent.postMessage('budget','*')">
            <div class="bubble-icon">📊</div>
            <div class="bubble-label">Budget</div>
            <div class="bubble-sub">50/30/20</div>
        </div>

        <div class="bubble b3 float-anim" onclick="window.parent.postMessage('metrics','*')">
            <div class="bubble-icon">📈</div>
            <div class="bubble-label">Metrics</div>
            <div class="bubble-sub">Key ratios</div>
        </div>

        <div class="bubble b4 float-anim" onclick="window.parent.postMessage('spending','*')">
            <div class="bubble-icon">🛒</div>
            <div class="bubble-label">Spending</div>
            <div class="bubble-sub">Categories</div>
        </div>

        <div class="bubble b5 float-anim" onclick="window.parent.postMessage('projection','*')">
            <div class="bubble-icon">🔮</div>
            <div class="bubble-label">Future</div>
            <div class="bubble-sub">6 months</div>
        </div>

        <div class="bubble b6 float-anim" onclick="window.parent.postMessage('ai','*')">
            <div class="bubble-icon">🤖</div>
            <div class="bubble-label">AI Advisor</div>
            <div class="bubble-sub">Get tips</div>
        </div>

        <div class="hint">tap any bubble to explore</div>
    </div>
    """

    components.html(bubble_html, height=700)

    # Bubble selector buttons (hidden but functional)
    st.markdown("""
    <style>
    .bubble-btn-row { display:flex; gap:12px; flex-wrap:wrap; justify-content:center; padding:0 40px 20px; }
    .bubble-btn {
        background:#0d1117; border:1px solid #1e2535;
        border-radius:999px; padding:10px 20px;
        font-size:12px; font-weight:600; color:#888;
        cursor:pointer; transition:all 0.3s;
        text-transform:uppercase; letter-spacing:1px;
    }
    .bubble-btn:hover { border-color:#378ADD; color:#378ADD; }
    </style>
    <div class="bubble-btn-row">
        <span style="color:#2a3545;font-size:11px;text-transform:uppercase;letter-spacing:2px;width:100%;text-align:center;margin-bottom:8px;">
            Select a section to explore
        </span>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(6)
    bubble_labels = ["🎯 Goals", "📊 Budget", "📈 Metrics", "🛒 Spending", "🔮 Future", "🤖 AI"]
    bubble_keys   = ["goals", "budget", "metrics", "spending", "projection", "ai"]

    for i, (col, lbl, key) in enumerate(zip(cols, bubble_labels, bubble_keys)):
        with col:
            if st.button(lbl, key=f"bb_{key}"):
                st.session_state.active_bubble = key
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Active Bubble Content ─────────────────────
    active = st.session_state.active_bubble

    if active:
        # Transition animation
        components.html(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@700;800;900&display=swap');
        .panel-enter {{
            animation: panelIn 0.5s cubic-bezier(0.34,1.56,0.64,1) forwards;
        }}
        @keyframes panelIn {{
            from {{ opacity:0; transform:translateY(30px) scale(0.97); }}
            to   {{ opacity:1; transform:translateY(0) scale(1); }}
        }}
        .panel-header {{
            padding:0 40px 20px;
            font-family:'Inter',sans-serif;
        }}
        .panel-title {{
            font-size:32px; font-weight:900; color:#fff;
        }}
        .panel-sub {{
            font-size:13px; color:#4a5568; margin-top:6px;
            text-transform:uppercase; letter-spacing:2px;
        }}
        </style>
        <div class="panel-enter panel-header">
            <div class="panel-sub">Exploring</div>
            <div class="panel-title">{active.replace('_',' ').title()}</div>
        </div>
        """, height=100)

        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        if active == "goals":
            for i, gr in enumerate(goal_results):
                goal         = gr['goal']
                remaining    = gr['remaining']
                npm          = gr['needed_per_month']
                achievable   = gr['achievable']
                real_months  = gr['realistic_months']
                comp_date    = gr['completion_date']
                prog         = gr['progress_pct']
                bar_color    = "#2ecc71" if prog>=50 else "#f39c12" if prog>=25 else "#378ADD"
                target_date  = datetime.now() + relativedelta(months=int(goal['months']))

                components.html(f"""
                <style>
                @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
                * {{ font-family:'Inter',sans-serif; box-sizing:border-box; }}
                .gcard {{
                    background:#0d1117; border-radius:20px;
                    padding:28px; border:1px solid #1e2535;
                    margin:0 40px 20px; color:#fff;
                    animation: panelIn 0.5s ease forwards;
                }}
                @keyframes panelIn {{
                    from {{ opacity:0; transform:translateY(20px); }}
                    to   {{ opacity:1; transform:translateY(0); }}
                }}
                .gtop {{ display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:20px; }}
                .gname {{ font-size:22px; font-weight:800; }}
                .gpct {{ font-size:32px; font-weight:900; color:{bar_color}; }}
                .track {{ background:#131720; border-radius:999px; height:12px; overflow:hidden; margin:12px 0; }}
                .fill {{ height:100%; border-radius:999px; background:{bar_color}; width:{prog}%;
                         box-shadow:0 0 12px {bar_color}66; transition:width 1s ease; }}
                .stats {{ display:flex; gap:16px; margin-top:20px; flex-wrap:wrap; }}
                .stat {{ background:#131720; border-radius:12px; padding:14px 18px; flex:1; min-width:120px; }}
                .sv {{ font-size:20px; font-weight:800; color:#fff; }}
                .sl {{ font-size:10px; color:#4a5568; text-transform:uppercase; letter-spacing:1px; margin-top:4px; }}
                .status {{ margin-top:16px; padding:12px 16px; border-radius:12px;
                           background:{'#0d2a1a' if achievable else '#2a0d0d'};
                           border:1px solid {'#2ecc7133' if achievable else '#e74c3c33'};
                           color:{'#2ecc71' if achievable else '#e74c3c'};
                           font-size:13px; font-weight:600; }}
                </style>
                <div class="gcard">
                    <div class="gtop">
                        <div>
                            <div style="font-size:11px;color:#4a5568;text-transform:uppercase;letter-spacing:2px;margin-bottom:6px;">Goal {i+1}</div>
                            <div class="gname">🎯 {goal['name']}</div>
                        </div>
                        <div class="gpct">{prog:.0f}%</div>
                    </div>
                    <div class="track"><div class="fill"></div></div>
                    <div style="display:flex;justify-content:space-between;font-size:11px;color:#2a3545;">
                        <span>Saved: &#8377;{savings:,.0f}</span>
                        <span>Target: &#8377;{goal['cost']:,.0f}</span>
                    </div>
                    <div class="stats">
                        <div class="stat"><div class="sv">&#8377;{remaining:,.0f}</div><div class="sl">Remaining</div></div>
                        <div class="stat"><div class="sv">&#8377;{npm:,.0f}</div><div class="sl">Needed/Month</div></div>
                        <div class="stat"><div class="sv">&#8377;{cf:,.0f}</div><div class="sl">Your Surplus</div></div>
                        <div class="stat"><div class="sv">{comp_date.strftime('%b %Y')}</div><div class="sl">Est. Completion</div></div>
                    </div>
                    <div class="status">
                        {'✅ Achievable by ' + target_date.strftime('%B %Y') + ' with your current surplus!' if achievable
                         else '❌ Realistic timeline: ' + str(int(real_months)) + ' months by ' + comp_date.strftime('%B %Y')}
                    </div>
                </div>
                """, height=380)

                # Simulator
                st.markdown(f"**🔧 Savings Simulator for {goal['name']}**")
                extra = st.slider("If I save more per month (₹)", 0, int(income), 0, 500, key=f"sl{i}")
                if extra > 0:
                    new_months = remaining / (cf+extra) if (cf+extra)>0 else 9999
                    new_date   = datetime.now() + relativedelta(months=int(new_months))
                    saved_m    = max(0, real_months - new_months)
                    st.success(f"💡 Saving ₹{extra:,.0f} more → Goal by **{new_date.strftime('%B %Y')}** — {saved_m:.0f} months earlier!")

                # Plan table
                st.markdown("**📅 Month by Month Plan**")
                plan = []
                running = savings
                for m in range(1, min(int(goal['months'])+3, 13)):
                    ms  = min(npm, cf)
                    running += ms
                    pct = min(100, running/goal['cost']*100)
                    plan.append({'Month': f'Month {m}', 'Save': f'₹{ms:,.0f}',
                                 'Total': f'₹{running:,.0f}', 'Progress': f'{pct:.1f}%'})
                    if running >= goal['cost']: break
                st.dataframe(pd.DataFrame(plan), use_container_width=True, hide_index=True)

        elif active == "budget":
            exp_gap = expenses - income*0.5
            sav_gap = income*0.2 - savings
            ideal_lr  = float(np.clip(lr_model.predict([[income,income*0.5,income*0.2,debt]])[0],0,100))
            ideal_xgb = float(np.clip(xgb_model.predict([[income,income*0.5,income*0.2,debt]])[0],0,100))
            ideal_score = (ideal_lr+ideal_xgb)/2

            components.html(f"""
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
            * {{ font-family:'Inter',sans-serif; box-sizing:border-box; }}
            .bcard {{ background:#0d1117; border-radius:20px; padding:28px; border:1px solid #1e2535; margin:0 40px 20px; }}
            .brow {{ display:flex; justify-content:space-between; align-items:center; padding:14px 0; border-bottom:1px solid #131720; }}
            .brow:last-child {{ border:none; }}
            .blabel {{ font-size:15px; font-weight:600; color:#fff; }}
            .bsub {{ font-size:11px; color:#4a5568; margin-top:3px; text-transform:uppercase; letter-spacing:1px; }}
            .bval {{ text-align:right; }}
            .bamt {{ font-size:18px; font-weight:800; }}
            .byour {{ font-size:11px; margin-top:3px; }}
            </style>
            <div class="bcard">
                <div style="font-size:11px;color:#4a5568;text-transform:uppercase;letter-spacing:2px;margin-bottom:20px;">
                    Recommended split of &#8377;{income:,.0f}/month
                </div>
                <div class="brow">
                    <div><div class="blabel">Needs</div><div class="bsub">50% of income</div></div>
                    <div class="bval">
                        <div class="bamt" style="color:#378ADD">&#8377;{income*0.5:,.0f}</div>
                        <div class="byour" style="color:{'#2ecc71' if expenses<=income*0.5 else '#e74c3c'}">
                            You spend &#8377;{expenses:,.0f}
                        </div>
                    </div>
                </div>
                <div class="brow">
                    <div><div class="blabel">Wants</div><div class="bsub">30% of income</div></div>
                    <div class="bval"><div class="bamt" style="color:#f39c12">&#8377;{income*0.3:,.0f}</div></div>
                </div>
                <div class="brow">
                    <div><div class="blabel">Savings</div><div class="bsub">20% of income</div></div>
                    <div class="bval">
                        <div class="bamt" style="color:#2ecc71">&#8377;{income*0.2:,.0f}</div>
                        <div class="byour" style="color:{'#2ecc71' if savings>=income*0.2 else '#e74c3c'}">
                            You save &#8377;{savings:,.0f}
                        </div>
                    </div>
                </div>
            </div>
            """, height=280)

            c1, c2 = st.columns(2)
            with c1:
                if exp_gap > 0: st.error(f"❌ Reduce expenses by ₹{exp_gap:,.0f}")
                else:           st.success(f"✅ Expenses ₹{abs(exp_gap):,.0f} under the 50% limit")
                if sav_gap > 0: st.error(f"❌ Save ₹{sav_gap:,.0f} more to hit 20% target")
                else:           st.success(f"✅ Savings ₹{abs(sav_gap):,.0f} above 20% target")
                if ideal_score > final_score:
                    st.info(f"💡 Following 50/30/20 improves your score by {ideal_score-final_score:.1f} pts")
            with c2:
                fig, ax = plt.subplots(figsize=(5,3.5), facecolor='#0d1117')
                ax.set_facecolor('#0d1117')
                cats = ['Needs\n50%','Wants\n30%','Savings\n20%','Your\nExpenses','Your\nSavings']
                vals = [income*0.5,income*0.3,income*0.2,expenses,savings]
                clrs = ['#1a5fa8','#7a4f00','#1a6b3a','#8b1a1a','#1a6b3a']
                bars = ax.bar(cats,vals,color=clrs,edgecolor='#0d1117',width=0.55)
                for bar,val in zip(bars,vals):
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+50,
                            f'₹{val:,.0f}', ha='center', color='#888', fontsize=8)
                ax.tick_params(colors='#333',labelsize=9)
                for s in ['top','right','left']: ax.spines[s].set_visible(False)
                ax.spines['bottom'].set_color('#1a2035')
                ax.yaxis.set_visible(False)
                plt.tight_layout()
                st.pyplot(fig); plt.close()

        elif active == "metrics":
            components.html(f"""
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
            * {{ font-family:'Inter',sans-serif; box-sizing:border-box; }}
            .mgrid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; padding:0 40px 20px; }}
            .mcard {{
                background:#0d1117; border-radius:20px; padding:28px;
                border:1px solid #1e2535;
                animation: panelIn 0.5s ease forwards;
            }}
            @keyframes panelIn {{
                from {{ opacity:0; transform:translateY(20px); }}
                to   {{ opacity:1; transform:translateY(0); }}
            }}
            .mcard:nth-child(2) {{ animation-delay:0.1s; }}
            .mcard:nth-child(3) {{ animation-delay:0.2s; }}
            .mcard:nth-child(4) {{ animation-delay:0.3s; }}
            .micon {{ font-size:28px; margin-bottom:12px; }}
            .mval {{ font-size:36px; font-weight:900; line-height:1; }}
            .mlbl {{ font-size:11px; color:#4a5568; text-transform:uppercase; letter-spacing:1px; margin-top:8px; }}
            .mideal {{ font-size:11px; margin-top:6px; }}
            .mbar {{ background:#131720; border-radius:999px; height:4px; margin-top:12px; overflow:hidden; }}
            .mfill {{ height:100%; border-radius:999px; }}
            </style>
            <div class="mgrid">
                <div class="mcard">
                    <div class="micon">💸</div>
                    <div class="mval" style="color:{'#2ecc71' if er<=70 else '#e74c3c'}">{er:.1f}%</div>
                    <div class="mlbl">Expense Ratio</div>
                    <div class="mideal" style="color:#4a5568">Ideal: under 70%</div>
                    <div class="mbar"><div class="mfill" style="width:{min(100,er)}%;background:{'#2ecc71' if er<=70 else '#e74c3c'};"></div></div>
                </div>
                <div class="mcard">
                    <div class="micon">🏦</div>
                    <div class="mval" style="color:{'#2ecc71' if sr>=20 else '#e74c3c'}">{sr:.1f}%</div>
                    <div class="mlbl">Savings Rate</div>
                    <div class="mideal" style="color:#4a5568">Ideal: above 20%</div>
                    <div class="mbar"><div class="mfill" style="width:{min(100,sr)}%;background:{'#2ecc71' if sr>=20 else '#e74c3c'};"></div></div>
                </div>
                <div class="mcard">
                    <div class="micon">💳</div>
                    <div class="mval" style="color:{'#2ecc71' if dti<=36 else '#e74c3c'}">{dti:.1f}%</div>
                    <div class="mlbl">Debt to Income</div>
                    <div class="mideal" style="color:#4a5568">Ideal: under 36%</div>
                    <div class="mbar"><div class="mfill" style="width:{min(100,dti)}%;background:{'#2ecc71' if dti<=36 else '#e74c3c'};"></div></div>
                </div>
                <div class="mcard">
                    <div class="micon">💵</div>
                    <div class="mval" style="color:{'#2ecc71' if cf>=0 else '#e74c3c'}">&#8377;{cf:,.0f}</div>
                    <div class="mlbl">Monthly Surplus</div>
                    <div class="mideal" style="color:#4a5568">Income minus expenses</div>
                    <div class="mbar"><div class="mfill" style="width:{'50' if cf>=0 else '100'}%;background:{'#2ecc71' if cf>=0 else '#e74c3c'};"></div></div>
                </div>
            </div>
            """, height=420)

        elif active == "spending":
            cat_df = df[df['Transaction Type']=='debit'].groupby('Category')['Amount'].sum().sort_values(ascending=False).head(8)
            c1, c2 = st.columns(2)
            with c1:
                cd = cat_df.reset_index()
                cd.columns = ['Category','Spent']
                cd['Spent'] = cd['Spent'].apply(lambda x: f'₹{x:,.0f}')
                st.dataframe(cd, use_container_width=True, hide_index=True)
            with c2:
                fig, ax = plt.subplots(figsize=(5,3.5), facecolor='#0d1117')
                ax.set_facecolor('#0d1117')
                colors = plt.cm.Blues(np.linspace(0.35,0.85,len(cat_df)))
                cat_df.plot(kind='barh',ax=ax,color=colors,edgecolor='#0d1117')
                ax.tick_params(colors='#333',labelsize=9)
                for s in ['top','right']: ax.spines[s].set_visible(False)
                ax.spines['bottom'].set_color('#1a2035')
                ax.spines['left'].set_color('#1a2035')
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'₹{x:,.0f}'))
                ax.set_xlabel(''); ax.set_ylabel('')
                plt.tight_layout()
                st.pyplot(fig); plt.close()

        elif active == "projection":
            pe_vals = [expenses*(1-0.01*i) for i in range(1,7)]
            ps_vals = [savings *(1+0.01*i) for i in range(1,7)]
            ms      = [f'M{i}' for i in range(1,7)]
            c1, c2  = st.columns(2)
            with c1:
                rows = [{'Month':ms[i],'Expenses':f'₹{pe_vals[i]:,.0f}',
                         'Savings':f'₹{ps_vals[i]:,.0f}','Surplus':f'₹{income-pe_vals[i]:,.0f}'} for i in range(6)]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            with c2:
                fig, ax = plt.subplots(figsize=(5,3.5), facecolor='#0d1117')
                ax.set_facecolor('#0d1117')
                ax.plot(ms,pe_vals,marker='o',color='#e74c3c',linewidth=2.5,markersize=6,label='Expenses')
                ax.plot(ms,ps_vals,marker='s',color='#2ecc71',linewidth=2.5,markersize=6,label='Savings')
                ax.fill_between(ms,pe_vals,alpha=0.05,color='#e74c3c')
                ax.fill_between(ms,ps_vals,alpha=0.05,color='#2ecc71')
                ax.tick_params(colors='#333',labelsize=9)
                for s in ['top','right']: ax.spines[s].set_visible(False)
                ax.spines['bottom'].set_color('#1a2035')
                ax.spines['left'].set_color('#1a2035')
                ax.legend(facecolor='#0d1117',labelcolor='#888',fontsize=9,framealpha=0.5)
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'₹{x:,.0f}'))
                plt.tight_layout()
                st.pyplot(fig); plt.close()

        elif active == "ai":
            with st.spinner("AI is analysing your finances..."):
                try:
                    goals_text = "\n".join([
                        f"- {g['goal']['name']}: Rs {g['goal']['cost']:,.0f} in {g['goal']['months']:.0f} months, needs Rs {g['needed_per_month']:,.0f}/month, {'achievable' if g['achievable'] else 'NOT achievable in time'}"
                        for g in goal_results
                    ])
                    prompt = f"""
You are a friendly personal financial advisor for {name}.

FINANCES:
- Income: Rs {income:,.0f} | Expenses: Rs {expenses:,.0f} | Savings: Rs {savings:,.0f} | Debt: Rs {debt:,.0f}
- Monthly Surplus: Rs {cf:,.0f} | Score: {final_score:.1f}/100 ({label})

GOALS:
{goals_text}

TOP SPENDING: {top_3_cats[0]} (Rs {top_3_amounts[0]:,.0f}), {top_3_cats[1]} (Rs {top_3_amounts[1]:,.0f}), {top_3_cats[2]} (Rs {top_3_amounts[2]:,.0f})

Give {name} exactly 5 short friendly specific tips to achieve their goals faster.
Use rupee amounts. No jargon. No mention of ML or models. Talk like a friend.
"""
                    response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role":"user","content":prompt}]
                    )
                    st.markdown(response.choices[0].message.content)
                except Exception as e:
                    st.error(f"Could not load recommendations: {e}")

        st.markdown("<br>", unsafe_allow_html=True)

        # Back + Reset buttons
        c1, c2, c3 = st.columns([1,1,1])
        with c2:
            if st.button("← Back to Dashboard"):
                st.session_state.active_bubble = None
                st.rerun()

    # Reset
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2,1,2])
    with col2:
        if st.button("Start Over"):
            st.session_state.screen = 1
            st.session_state.active_bubble = None
            st.rerun()
