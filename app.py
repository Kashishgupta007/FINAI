import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from groq import Groq
from datetime import datetime
from dateutil.relativedelta import relativedelta
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

client = Groq(api_key="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")

st.set_page_config(page_title="FinAI", page_icon="💰", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');
* { font-family: 'Space Grotesk', sans-serif !important; margin: 0; padding: 0; box-sizing: border-box; }
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: #04060f !important; color: #fff;
}
[data-testid="stHeader"] { display: none; }
[data-testid="stSidebar"] { display: none; }
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stMain"] > div { padding: 0 !important; }
div[data-testid="stNumberInput"] > div > div > input,
div[data-testid="stTextInput"] > div > div > input {
    background: #0b0f1e !important; border: 1px solid #1a2040 !important;
    border-radius: 14px !important; color: #fff !important;
    font-size: 15px !important; padding: 14px 16px !important;
    transition: all 0.3s ease !important;
}
div[data-testid="stNumberInput"] > div > div > input:focus,
div[data-testid="stTextInput"] > div > div > input:focus {
    border-color: #4f8ef7 !important;
    box-shadow: 0 0 0 3px rgba(79,142,247,0.15) !important;
}
div[data-testid="stNumberInput"] label,
div[data-testid="stTextInput"] label,
div[data-testid="stSelectbox"] label {
    color: #3a4560 !important; font-size: 10px !important;
    text-transform: uppercase !important; letter-spacing: 2px !important; font-weight: 600 !important;
}
div[data-testid="stSelectbox"] > div > div {
    background: #0b0f1e !important; border: 1px solid #1a2040 !important;
    border-radius: 14px !important; color: #fff !important;
}
.stButton > button {
    background: linear-gradient(135deg, #1a4fa8 0%, #4f8ef7 100%) !important;
    color: white !important; border: none !important; border-radius: 14px !important;
    padding: 16px 40px !important; font-size: 15px !important; font-weight: 700 !important;
    letter-spacing: 0.5px !important; width: 100% !important; transition: all 0.3s ease !important;
    box-shadow: 0 8px 32px rgba(79,142,247,0.25) !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 14px 40px rgba(79,142,247,0.4) !important; }
[data-testid="stRadio"] label { color: #fff !important; }
div[data-testid="stMetric"] {
    background: #0b0f1e !important; border: 1px solid #1a2040 !important;
    border-radius: 16px !important; padding: 20px !important;
}
div[data-testid="stMetric"] label { color: #3a4560 !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────
for key, val in [
    ('screen', 1), ('user_data', {}), ('finance_data', {}),
    ('active_page', None), ('chat_history', []),
]:
    if key not in st.session_state:
        st.session_state[key] = val

# ── Data & Models ─────────────────────────────────
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
    X = np.array(X_tuple); y = np.array(y_tuple)
    lr = LinearRegression(); lr.fit(X, y)
    xgb = XGBRegressor(n_estimators=100, random_state=42, verbosity=0); xgb.fit(X, y)
    return lr, xgb

df      = load_default_data()
monthly = prepare_monthly(df)
X       = monthly[['income','expenses','savings','debt']].values
y       = monthly['health_score'].values
lr_model, xgb_model = train_models(tuple(map(tuple,X)), tuple(y))
cat_spending  = df[df['Transaction Type']=='debit'].groupby('Category')['Amount'].sum().sort_values(ascending=False)
top_3_cats    = list(cat_spending.head(3).index)
top_3_amounts = list(cat_spending.head(3).values)

def get_label(s):
    if s>=75:  return "Excellent","#00e5a0"
    elif s>=50: return "Good","#4f8ef7"
    elif s>=30: return "Fair","#f7c94f"
    else:       return "Poor","#f74f4f"

def dark_chart(ax,fig):
    fig.patch.set_facecolor('#0b0f1e'); ax.set_facecolor('#0b0f1e')
    ax.tick_params(colors='#3a4560',labelsize=9)
    for sp in ax.spines.values(): sp.set_color('#1a2040')

def page_header(icon,title,subtitle,color="#4f8ef7"):
    components.html(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@800&family=Space+Grotesk:wght@400;600&display=swap');
    .ph{{padding:36px 48px 24px;border-bottom:1px solid #0f1528;}}
    .ph-icon{{font-size:34px;margin-bottom:10px;}}
    .ph-title{{font-family:'Syne',sans-serif;font-size:40px;font-weight:800;color:{color};line-height:1;}}
    .ph-sub{{font-size:11px;color:#3a4560;margin-top:10px;letter-spacing:2px;text-transform:uppercase;}}
    .ph-line{{width:44px;height:3px;background:{color};border-radius:99px;margin-top:14px;}}
    </style>
    <div class="ph"><div class="ph-icon">{icon}</div>
    <div class="ph-title">{title}</div>
    <div class="ph-sub">{subtitle}</div>
    <div class="ph-line"></div></div>
    """, height=190)

def back_button():
    st.markdown("<br>", unsafe_allow_html=True)
    c1,c2,c3 = st.columns([1,1,1])
    with c2:
        if st.button("← Back to Dashboard", key="back_btn_main"):
            st.session_state.active_page = None
            st.rerun()

def bottom_nav(active):
    st.markdown("---")
    st.markdown("<p style='text-align:center;color:#3a4560;font-size:10px;text-transform:uppercase;letter-spacing:2px;margin-bottom:8px;'>Quick Navigate</p>",unsafe_allow_html=True)
    pages  = ["goals","budget","metrics","spending","emergency","projection","networth","ai"]
    labels = ["🎯 Goals","📊 Budget","📈 Metrics","🛒 Spending","🛡️ Emergency","🔮 Future","💎 Net Worth","🤖 AI"]
    cols   = st.columns(8)
    for col,lbl,pg in zip(cols,labels,pages):
        with col:
            if st.button(lbl, key=f"bnav_{pg}"):
                st.session_state.active_page = pg
                st.rerun()

# ════════════════════════════════════════════════
# SCREEN 1 — WELCOME
# ════════════════════════════════════════════════
if st.session_state.screen == 1:
    components.html("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Space+Grotesk:wght@300;400;500;600&display=swap');
    *{margin:0;padding:0;box-sizing:border-box;}
    body{background:#04060f;overflow:hidden;}
    .bg{position:fixed;inset:0;
        background:radial-gradient(ellipse at 20% 50%,rgba(79,142,247,0.07) 0%,transparent 60%),
                   radial-gradient(ellipse at 80% 20%,rgba(0,229,160,0.05) 0%,transparent 50%);}
    .grid{position:fixed;inset:0;
        background-image:linear-gradient(rgba(255,255,255,0.018) 1px,transparent 1px),
                         linear-gradient(90deg,rgba(255,255,255,0.018) 1px,transparent 1px);
        background-size:80px 80px;pointer-events:none;}
    .wrap{position:relative;z-index:10;min-height:100vh;display:flex;flex-direction:column;
          align-items:center;justify-content:center;padding:60px 20px;}
    .badge{display:inline-flex;align-items:center;gap:10px;
           background:rgba(79,142,247,0.08);border:1px solid rgba(79,142,247,0.18);
           border-radius:999px;padding:8px 22px;font-size:11px;font-weight:600;
           color:#4f8ef7;letter-spacing:2.5px;text-transform:uppercase;margin-bottom:36px;
           animation:fadeUp 0.8s ease forwards;font-family:'Space Grotesk',sans-serif;}
    .dot{width:6px;height:6px;background:#00e5a0;border-radius:50%;animation:pulse 2s infinite;}
    @keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:0.4;transform:scale(1.8)}}
    .title{font-family:'Syne',sans-serif;font-size:clamp(52px,9vw,96px);font-weight:800;
           line-height:0.95;text-align:center;margin-bottom:24px;animation:fadeUp 0.8s ease 0.15s both;}
    .l1{color:#fff;display:block;}
    .l2{display:block;background:linear-gradient(90deg,#4f8ef7 0%,#00e5a0 100%);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;}
    .sub{font-size:17px;color:#3a4560;text-align:center;max-width:460px;line-height:1.7;
         margin-bottom:52px;animation:fadeUp 0.8s ease 0.3s both;font-family:'Space Grotesk',sans-serif;}
    .stats{display:flex;gap:48px;margin-bottom:52px;animation:fadeUp 0.8s ease 0.45s both;}
    .stat{text-align:center;}
    .sn{font-family:'Syne',sans-serif;font-size:32px;font-weight:800;color:#fff;}
    .sn span{color:#4f8ef7;}
    .sl{font-size:10px;color:#3a4560;text-transform:uppercase;letter-spacing:1.5px;margin-top:6px;font-family:'Space Grotesk',sans-serif;}
    .div{width:1px;height:36px;background:#1a2040;align-self:center;}
    .scroll{display:flex;flex-direction:column;align-items:center;gap:10px;color:#1a2040;
            font-size:10px;letter-spacing:3px;text-transform:uppercase;font-family:'Space Grotesk',sans-serif;
            animation:fadeUp 0.8s ease 0.6s both;}
    .arr{font-size:18px;animation:bounce 2s infinite;color:#1a2040;}
    @keyframes bounce{0%,100%{transform:translateY(0)}50%{transform:translateY(10px)}}
    @keyframes fadeUp{from{opacity:0;transform:translateY(28px)}to{opacity:1;transform:translateY(0)}}
    .o1{position:fixed;width:500px;height:500px;background:rgba(79,142,247,0.06);top:-200px;right:-100px;border-radius:50%;filter:blur(100px);}
    .o2{position:fixed;width:400px;height:400px;background:rgba(0,229,160,0.04);bottom:-150px;left:-100px;border-radius:50%;filter:blur(100px);}
    </style>
    <div class="bg"></div><div class="grid"></div><div class="o1"></div><div class="o2"></div>
    <div class="wrap">
        <div class="badge"><div class="dot"></div>AI-Powered · ML-Driven · Personalised</div>
        <h1 class="title"><span class="l1">Know Your</span><span class="l2">Financial Health</span></h1>
        <p class="sub">Answer a few questions. Get a personalised financial health score, goal tracker, and an AI-powered plan — built just for you.</p>
        <div class="stats">
            <div class="stat"><div class="sn">806<span>+</span></div><div class="sl">Transactions</div></div>
            <div class="div"></div>
            <div class="stat"><div class="sn">21</div><div class="sl">Months Data</div></div>
            <div class="div"></div>
            <div class="stat"><div class="sn">2</div><div class="sl">ML Models</div></div>
            <div class="div"></div>
            <div class="stat"><div class="sn">8</div><div class="sl">Insight Panels</div></div>
        </div>
        <div class="scroll"><span>scroll to begin</span><div class="arr">↓</div></div>
    </div>
    """, height=620)

    st.markdown("<br>", unsafe_allow_html=True)
    col1,col2,col3 = st.columns([1,2,1])
    with col2:
        components.html("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@800&family=Space+Grotesk:wght@400;600&display=swap');
        .sh{text-align:center;margin-bottom:28px;}
        .sh-pre{color:#3a4560;font-size:10px;text-transform:uppercase;letter-spacing:3px;font-family:'Space Grotesk',sans-serif;}
        .sh-t{font-family:'Syne',sans-serif;font-size:28px;font-weight:800;color:#fff;margin-top:8px;}
        </style>
        <div class="sh"><div class="sh-pre">Step 1 of 2</div><div class="sh-t">Tell us about yourself</div></div>
        """, height=88)
        name = st.text_input("Your Name", placeholder="e.g. Kashish Gupta")
        age  = st.number_input("Your Age", min_value=18, max_value=80, value=22, step=1)
        occ  = st.selectbox("Occupation", ["Student","Salaried Employee","Self Employed","Freelancer","Business Owner","Other"])
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Continue →"):
            if name.strip() == "":
                st.error("Please enter your name to continue")
            else:
                st.session_state.user_data = {"name":name.strip(),"age":age,"occupation":occ}
                st.session_state.screen = 2
                st.rerun()

# ════════════════════════════════════════════════
# SCREEN 2 — FINANCIAL DATA
# ════════════════════════════════════════════════
elif st.session_state.screen == 2:
    name = st.session_state.user_data.get("name","there")
    components.html(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@800&family=Space+Grotesk:wght@400;600&display=swap');
    *{{margin:0;padding:0;box-sizing:border-box;}}
    body{{background:#04060f;}}
    .bg{{position:fixed;inset:0;background:radial-gradient(ellipse at 75% 25%,rgba(79,142,247,0.07) 0%,transparent 55%);}}
    .grid{{position:fixed;inset:0;
        background-image:linear-gradient(rgba(255,255,255,0.015) 1px,transparent 1px),
                         linear-gradient(90deg,rgba(255,255,255,0.015) 1px,transparent 1px);
        background-size:80px 80px;pointer-events:none;}}
    .hdr{{position:relative;z-index:10;padding:40px 48px 24px;border-bottom:1px solid #0f1528;}}
    .hdr-pre{{font-size:11px;color:#4f8ef7;text-transform:uppercase;letter-spacing:3px;font-weight:600;margin-bottom:8px;font-family:'Space Grotesk',sans-serif;}}
    .hdr-title{{font-family:'Syne',sans-serif;font-size:38px;font-weight:800;color:#fff;}}
    .hdr-name{{color:#00e5a0;}}
    .steps{{display:flex;gap:6px;margin-top:18px;}}
    .s{{width:32px;height:4px;border-radius:99px;background:#1a2040;}}
    .s.done{{background:#4f8ef7;}}
    .s.active{{background:#00e5a0;box-shadow:0 0 10px rgba(0,229,160,0.5);}}
    </style>
    <div class="bg"></div><div class="grid"></div>
    <div class="hdr">
        <div class="hdr-pre">Welcome back, {name}</div>
        <div class="hdr-title">Your <span class="hdr-name">Financial</span> Details</div>
        <div class="steps"><div class="s done"></div><div class="s active"></div><div class="s"></div></div>
    </div>
    """, height=178)

    col1,col2,col3 = st.columns([1,3,1])
    with col2:
        components.html("""<div style="padding:24px 0 8px;"><div style="font-size:10px;color:#4f8ef7;text-transform:uppercase;letter-spacing:3px;font-weight:600;font-family:'Space Grotesk',sans-serif;">Monthly Finances</div></div>""", height=54)
        c1,c2 = st.columns(2)
        with c1:
            income   = st.number_input("Monthly Income (₹)",   min_value=0.0, value=50000.0,  step=1000.0)
            savings  = st.number_input("Monthly Savings (₹)",  min_value=0.0, value=10000.0,  step=500.0)
        with c2:
            expenses = st.number_input("Monthly Expenses (₹)", min_value=0.0, value=30000.0,  step=1000.0)
            debt     = st.number_input("Monthly Debt/EMI (₹)", min_value=0.0, value=5000.0,   step=500.0)

        components.html("""<div style="padding:20px 0 8px;"><div style="font-size:10px;color:#00e5a0;text-transform:uppercase;letter-spacing:3px;font-weight:600;font-family:'Space Grotesk',sans-serif;">Emergency Fund</div></div>""", height=50)
        emergency_savings = st.number_input("Current Emergency Fund (₹)", min_value=0.0, value=20000.0, step=1000.0)
        emergency_months  = st.slider("Target: months of expenses covered", 3, 12, 6)

        components.html("""<div style="padding:20px 0 8px;"><div style="font-size:10px;color:#a78bfa;text-transform:uppercase;letter-spacing:3px;font-weight:600;font-family:'Space Grotesk',sans-serif;">Your Goals</div></div>""", height=50)
        num_goals = st.radio("", ["1 Goal","2 Goals","3 Goals"], horizontal=True)
        n_goals   = int(num_goals[0])

        goals = []
        gcols = st.columns(n_goals)
        defs  = [("Buy a Car",1500000.0,24.0),("Europe Trip",200000.0,12.0),("Emergency Fund",300000.0,18.0)]
        for i in range(n_goals):
            with gcols[i]:
                components.html(f"""<div style="background:#0b0f1e;border-radius:14px;padding:14px 16px;border:1px solid #1a2040;margin-bottom:8px;"><div style="font-size:10px;color:#a78bfa;font-weight:700;text-transform:uppercase;letter-spacing:2px;font-family:'Space Grotesk',sans-serif;">Goal {i+1}</div></div>""", height=56)
                n = st.text_input("Goal name",  value=defs[i][0], key=f"gn{i}")
                c = st.number_input("Cost (₹)", min_value=0.0, value=defs[i][1], step=10000.0, key=f"gc{i}")
                m = st.number_input("In months",min_value=1.0, value=defs[i][2], step=1.0, key=f"gm{i}")
                goals.append({"name":n,"cost":c,"months":m})

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Analyse My Finances ✦"):
            st.session_state.finance_data = {
                "income":income,"expenses":expenses,"savings":savings,"debt":debt,
                "emergency_savings":emergency_savings,"emergency_months":emergency_months,"goals":goals
            }
            st.session_state.screen       = 3
            st.session_state.active_page  = None
            st.rerun()

# ════════════════════════════════════════════════
# SCREEN 3 — DASHBOARD + PAGES
# ════════════════════════════════════════════════
elif st.session_state.screen == 3:
    fd   = st.session_state.finance_data
    ud   = st.session_state.user_data
    name = ud.get("name","there")
    age  = ud.get("age",25)
    occ  = ud.get("occupation","")

    income            = fd['income']
    expenses          = fd['expenses']
    savings           = fd['savings']
    debt              = fd['debt']
    emergency_savings = fd.get('emergency_savings',0)
    emergency_months  = fd.get('emergency_months',6)
    goals             = fd['goals']
    cf                = income - expenses

    # Score — model trained with debt=expenses*0.10; we add a real DTI penalty
    model_debt  = expenses * 0.10
    lr_score    = float(np.clip(lr_model.predict([[income,expenses,savings,model_debt]])[0],0,100))
    xgb_score   = float(np.clip(xgb_model.predict([[income,expenses,savings,model_debt]])[0],0,100))
    dti         = debt/income if income>0 else 0
    debt_pen    = max(0,(dti-0.15)*40)
    final_score = max(0,min(100,(lr_score+xgb_score)/2 - debt_pen))
    label,hex_color = get_label(final_score)

    er      = expenses/income*100 if income>0 else 0
    sr      = savings/income*100  if income>0 else 0
    dti_pct = debt/income*100     if income>0 else 0
    ef_target = expenses * emergency_months
    ef_pct    = min(100, emergency_savings/ef_target*100) if ef_target>0 else 0

    goal_results = []
    for goal in goals:
        remaining  = max(0, goal['cost']-savings)
        npm        = remaining/goal['months'] if goal['months']>0 else remaining
        achievable = cf>=npm
        real_m     = remaining/cf if cf>0 else 9999
        comp_date  = datetime.now()+relativedelta(months=int(min(real_m,999)))
        prog       = min(100,savings/goal['cost']*100) if goal['cost']>0 else 0
        goal_results.append({"goal":goal,"remaining":remaining,"needed_per_month":npm,
                              "achievable":achievable,"realistic_months":real_m,
                              "completion_date":comp_date,"progress_pct":prog})

    # ─────────────────────────────────────────────
    # DASHBOARD (no page selected)
    # ─────────────────────────────────────────────
    if st.session_state.active_page is None:
        components.html(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
        *{{margin:0;padding:0;box-sizing:border-box;font-family:'Space Grotesk',sans-serif;}}
        body{{background:#04060f;overflow-x:hidden;}}
        .bg{{position:fixed;inset:0;
            background:radial-gradient(ellipse at 50% 40%,rgba(79,142,247,0.07) 0%,transparent 65%),
                       radial-gradient(ellipse at 20% 80%,rgba(0,229,160,0.04) 0%,transparent 50%);}}
        .grid{{position:fixed;inset:0;
            background-image:linear-gradient(rgba(255,255,255,0.013) 1px,transparent 1px),
                             linear-gradient(90deg,rgba(255,255,255,0.013) 1px,transparent 1px);
            background-size:80px 80px;pointer-events:none;}}
        .nav{{position:relative;z-index:100;display:flex;justify-content:space-between;align-items:center;
              padding:20px 48px;border-bottom:1px solid #0f1528;}}
        .logo{{font-family:'Syne',sans-serif;font-size:22px;font-weight:800;color:#fff;}}
        .logo span{{color:#4f8ef7;}}
        .pills{{display:flex;gap:10px;align-items:center;}}
        .pill{{background:#0b0f1e;border:1px solid #1a2040;border-radius:999px;
               padding:8px 16px;font-size:12px;color:#3a4560;font-weight:500;}}
        .spill{{background:rgba(79,142,247,0.08);border:1px solid rgba(79,142,247,0.2);
                border-radius:999px;padding:8px 16px;font-size:12px;color:#4f8ef7;font-weight:700;}}
        .stats-bar{{position:relative;z-index:10;display:flex;gap:0;
                    border-top:1px solid #0f1528;border-bottom:1px solid #0f1528;}}
        .si{{flex:1;padding:16px 24px;border-right:1px solid #0f1528;text-align:center;}}
        .si:last-child{{border:none;}}
        .sv{{font-family:'Syne',sans-serif;font-size:20px;font-weight:700;color:#fff;}}
        .sl{{font-size:9px;color:#3a4560;text-transform:uppercase;letter-spacing:2px;margin-top:4px;}}
        .hero{{position:relative;z-index:10;display:flex;flex-direction:column;
               align-items:center;justify-content:center;padding:44px 48px 28px;text-align:center;}}
        .hero-pre{{font-size:10px;color:#3a4560;text-transform:uppercase;letter-spacing:3px;margin-bottom:20px;}}
        .sw{{position:relative;width:230px;height:230px;display:flex;align-items:center;justify-content:center;margin-bottom:20px;}}
        .ro{{position:absolute;inset:0;border-radius:50%;border:1px solid {hex_color}16;animation:rp 4s ease-in-out infinite;}}
        .rm{{position:absolute;inset:14px;border-radius:50%;border:1px solid {hex_color}26;animation:rp 4s ease-in-out infinite 1s;}}
        @keyframes rp{{0%,100%{{transform:scale(1);opacity:0.6}}50%{{transform:scale(1.04);opacity:1}}}}
        .sc{{width:172px;height:172px;border-radius:50%;
             background:radial-gradient(circle at 35% 35%,#111827 0%,#04060f 100%);
             border:2px solid {hex_color}38;
             display:flex;flex-direction:column;align-items:center;justify-content:center;
             box-shadow:0 0 80px {hex_color}16,0 0 0 8px {hex_color}05;
             animation:sp 1s cubic-bezier(0.34,1.56,0.64,1) forwards;}}
        @keyframes sp{{from{{transform:scale(0.4);opacity:0}}to{{transform:scale(1);opacity:1}}}}
        .snum{{font-family:'Syne',sans-serif;font-size:54px;font-weight:800;color:{hex_color};line-height:1;}}
        .sof{{font-size:11px;color:#1a2040;margin-top:2px;}}
        .slbl{{font-size:13px;font-weight:700;color:{hex_color};margin-top:6px;letter-spacing:1px;}}
        .hero-sub{{font-size:13px;color:#3a4560;}}
        .bubbles{{position:relative;z-index:10;display:flex;flex-wrap:wrap;justify-content:center;
                  gap:18px;padding:28px 48px 18px;max-width:920px;margin:0 auto;}}
        .bubble{{width:126px;height:126px;border-radius:50%;display:flex;flex-direction:column;
                 align-items:center;justify-content:center;gap:5px;cursor:pointer;
                 transition:all 0.35s cubic-bezier(0.34,1.56,0.64,1);border:1px solid transparent;
                 animation:bp 0.7s cubic-bezier(0.34,1.56,0.64,1) both;}}
        .bubble:hover{{transform:scale(1.13) translateY(-6px);box-shadow:0 22px 56px rgba(0,0,0,0.6);}}
        .bubble:active{{transform:scale(0.95);}}
        .bi{{font-size:24px;line-height:1;}}
        .bl{{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:1px;text-align:center;}}
        .bs{{font-size:9px;opacity:0.55;text-align:center;}}
        .b1{{background:linear-gradient(145deg,#091a35,#0f2a55);border-color:#4f8ef722;color:#4f8ef7;animation-delay:0.1s;}}
        .b2{{background:linear-gradient(145deg,#1a0935,#2a0f55);border-color:#a78bfa22;color:#a78bfa;animation-delay:0.2s;}}
        .b3{{background:linear-gradient(145deg,#09350f,#0f551a);border-color:#00e5a022;color:#00e5a0;animation-delay:0.3s;}}
        .b4{{background:linear-gradient(145deg,#35280a,#55400f);border-color:#f7c94f22;color:#f7c94f;animation-delay:0.4s;}}
        .b5{{background:linear-gradient(145deg,#350909,#550f0f);border-color:#f74f4f22;color:#f74f4f;animation-delay:0.5s;}}
        .b6{{background:linear-gradient(145deg,#093535,#0f5555);border-color:#38bdf822;color:#38bdf8;animation-delay:0.6s;}}
        .b7{{background:linear-gradient(145deg,#200935,#380f55);border-color:#e879f922;color:#e879f9;animation-delay:0.7s;}}
        .b8{{background:linear-gradient(145deg,#0a2535,#0f3555);border-color:#00d4ff22;color:#00d4ff;animation-delay:0.8s;}}
        @keyframes bp{{from{{opacity:0;transform:scale(0.3) translateY(20px)}}to{{opacity:1;transform:scale(1) translateY(0)}}}}
        .hint{{text-align:center;padding:6px 0 22px;font-size:10px;color:#1a2040;
               text-transform:uppercase;letter-spacing:3px;animation:fi 1s ease 1.2s both;}}
        @keyframes fi{{from{{opacity:0}}to{{opacity:1}}}}
        </style>
        <div class="bg"></div><div class="grid"></div>
        <div class="nav">
            <div class="logo">Fin<span>AI</span></div>
            <div class="pills">
                <div class="pill">👤 {name} · {occ}</div>
                <div class="spill">Score: {final_score:.0f}/100</div>
            </div>
        </div>
        <div class="stats-bar">
            <div class="si"><div class="sv">₹{income:,.0f}</div><div class="sl">Income</div></div>
            <div class="si"><div class="sv" style="color:{'#f74f4f' if expenses>income*0.7 else '#00e5a0'}">₹{expenses:,.0f}</div><div class="sl">Expenses</div></div>
            <div class="si"><div class="sv" style="color:{'#00e5a0' if cf>=0 else '#f74f4f'}">₹{cf:,.0f}</div><div class="sl">Surplus</div></div>
            <div class="si"><div class="sv">₹{savings:,.0f}</div><div class="sl">Savings</div></div>
            <div class="si"><div class="sv" style="color:{'#f74f4f' if dti_pct>36 else '#f7c94f' if dti_pct>15 else '#00e5a0'}">{dti_pct:.0f}%</div><div class="sl">Debt/Income</div></div>
        </div>
        <div class="hero">
            <div class="hero-pre">Your Financial Health Score</div>
            <div class="sw">
                <div class="ro"></div><div class="rm"></div>
                <div class="sc">
                    <div class="snum">{final_score:.0f}</div>
                    <div class="sof">out of 100</div>
                    <div class="slbl">{label}</div>
                </div>
            </div>
            <div class="hero-sub">Tap any bubble below to explore a deeper insight</div>
        </div>
        <div class="bubbles">
            <div class="bubble b1" onclick="window.parent.postMessage({{type:'finai',page:'goals'}},'*')">
                <div class="bi">🎯</div><div class="bl">Goals</div><div class="bs">{len(goals)} active</div></div>
            <div class="bubble b2" onclick="window.parent.postMessage({{type:'finai',page:'budget'}},'*')">
                <div class="bi">📊</div><div class="bl">Budget</div><div class="bs">50/30/20</div></div>
            <div class="bubble b3" onclick="window.parent.postMessage({{type:'finai',page:'metrics'}},'*')">
                <div class="bi">📈</div><div class="bl">Metrics</div><div class="bs">Key ratios</div></div>
            <div class="bubble b4" onclick="window.parent.postMessage({{type:'finai',page:'spending'}},'*')">
                <div class="bi">🛒</div><div class="bl">Spending</div><div class="bs">Categories</div></div>
            <div class="bubble b5" onclick="window.parent.postMessage({{type:'finai',page:'emergency'}},'*')">
                <div class="bi">🛡️</div><div class="bl">Emergency</div><div class="bs">{ef_pct:.0f}% funded</div></div>
            <div class="bubble b6" onclick="window.parent.postMessage({{type:'finai',page:'projection'}},'*')">
                <div class="bi">🔮</div><div class="bl">Future</div><div class="bs">6 months</div></div>
            <div class="bubble b7" onclick="window.parent.postMessage({{type:'finai',page:'networth'}},'*')">
                <div class="bi">💎</div><div class="bl">Net Worth</div><div class="bs">Tracker</div></div>
            <div class="bubble b8" onclick="window.parent.postMessage({{type:'finai',page:'ai'}},'*')">
                <div class="bi">🤖</div><div class="bl">AI Advisor</div><div class="bs">Get tips</div></div>
        </div>
        <div class="hint">tap any bubble to explore</div>
        """, height=930)

        # Streamlit buttons — styled as subtle chips, act as real navigation
        st.markdown("""
        <style>
        .nav-row .stButton>button{
            background:transparent!important;border:1px solid #1a2040!important;
            border-radius:999px!important;padding:8px 16px!important;font-size:11px!important;
            font-weight:600!important;color:#3a4560!important;width:auto!important;
            box-shadow:none!important;letter-spacing:1px!important;text-transform:uppercase!important;
        }
        .nav-row .stButton>button:hover{border-color:#4f8ef7!important;color:#4f8ef7!important;transform:none!important;box-shadow:none!important;}
        </style>
        """, unsafe_allow_html=True)

        st.markdown('<div class="nav-row">', unsafe_allow_html=True)
        pages  = ["goals","budget","metrics","spending","emergency","projection","networth","ai"]
        labels = ["🎯 Goals","📊 Budget","📈 Metrics","🛒 Spending","🛡️ Emergency","🔮 Future","💎 Net Worth","🤖 AI"]
        cols   = st.columns(8)
        for col,lbl,pg in zip(cols,labels,pages):
            with col:
                if st.button(lbl, key=f"nav_{pg}"):
                    st.session_state.active_page = pg
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1,c2,c3 = st.columns([3,1,3])
        with c2:
            if st.button("↺ Start Over"):
                st.session_state.screen=1; st.session_state.active_page=None; st.rerun()

    # ─────────────────────────────────────────────
    # INDIVIDUAL PAGES
    # ─────────────────────────────────────────────
    else:
        active = st.session_state.active_page

        # ══ GOALS ═══════════════════════════════
        if active == "goals":
            page_header("🎯","Goal Tracker",f"{len(goals)} goals · personalised timeline","#a78bfa")

            for i,gr in enumerate(goal_results):
                goal,remaining,npm = gr['goal'],gr['remaining'],gr['needed_per_month']
                achievable,real_m  = gr['achievable'],gr['realistic_months']
                comp_date,prog     = gr['completion_date'],gr['progress_pct']
                bc = "#00e5a0" if prog>=50 else "#f7c94f" if prog>=25 else "#4f8ef7"
                td = datetime.now()+relativedelta(months=int(goal['months']))

                components.html(f"""
                <style>
                @import url('https://fonts.googleapis.com/css2?family=Syne:wght@800&family=Space+Grotesk:wght@400;600;700&display=swap');
                *{{font-family:'Space Grotesk',sans-serif;box-sizing:border-box;}}
                .gc{{background:#0b0f1e;border-radius:20px;padding:30px;border:1px solid #1a2040;
                      margin:0 48px 16px;color:#fff;animation:si 0.5s cubic-bezier(0.34,1.56,0.64,1) {i*0.1}s both;}}
                @keyframes si{{from{{opacity:0;transform:translateY(24px)}}to{{opacity:1;transform:translateY(0)}}}}
                .gt{{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:22px;}}
                .gnum{{font-size:10px;color:#3a4560;text-transform:uppercase;letter-spacing:2px;margin-bottom:6px;}}
                .gname{{font-family:'Syne',sans-serif;font-size:24px;font-weight:800;color:#fff;}}
                .gpct{{font-family:'Syne',sans-serif;font-size:38px;font-weight:800;color:{bc};}}
                .track{{background:#0f1528;border-radius:999px;height:10px;overflow:hidden;margin:12px 0 6px;}}
                .fill{{height:100%;border-radius:999px;background:linear-gradient(90deg,{bc}77,{bc});
                        width:{prog:.0f}%;box-shadow:0 0 16px {bc}44;}}
                .tm{{display:flex;justify-content:space-between;font-size:10px;color:#3a4560;}}
                .gstats{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:18px;}}
                .gs{{background:#0f1528;border-radius:12px;padding:14px;text-align:center;}}
                .gsv{{font-family:'Syne',sans-serif;font-size:17px;font-weight:700;color:#fff;}}
                .gsl{{font-size:9px;color:#3a4560;text-transform:uppercase;letter-spacing:1px;margin-top:4px;}}
                .status{{margin-top:14px;padding:13px 16px;border-radius:12px;
                          background:{'#091f14' if achievable else '#1f0909'};
                          border:1px solid {'#00e5a025' if achievable else '#f74f4f25'};
                          color:{'#00e5a0' if achievable else '#f74f4f'};font-size:13px;font-weight:600;}}
                </style>
                <div class="gc">
                    <div class="gt">
                        <div><div class="gnum">Goal {i+1}</div><div class="gname">{goal['name']}</div></div>
                        <div class="gpct">{prog:.0f}%</div>
                    </div>
                    <div class="track"><div class="fill"></div></div>
                    <div class="tm"><span>₹{savings:,.0f} saved</span><span>₹{goal['cost']:,.0f} target</span></div>
                    <div class="gstats">
                        <div class="gs"><div class="gsv">₹{remaining:,.0f}</div><div class="gsl">Remaining</div></div>
                        <div class="gs"><div class="gsv">₹{npm:,.0f}</div><div class="gsl">Need/Month</div></div>
                        <div class="gs"><div class="gsv">₹{cf:,.0f}</div><div class="gsl">Surplus</div></div>
                        <div class="gs"><div class="gsv">{comp_date.strftime('%b %Y')}</div><div class="gsl">Est. Done</div></div>
                    </div>
                    <div class="status">
                        {'✅  On track — achievable by ' + td.strftime('%B %Y') if achievable
                         else '⚠️  Will take ~' + str(int(real_m)) + ' months. Target: ' + td.strftime('%B %Y')}
                    </div>
                </div>
                """, height=370)

                c1,c2 = st.columns([2,1])
                with c1:
                    st.markdown(f"**🔧 Savings Simulator — {goal['name']}**")
                    extra = st.slider("Extra savings per month (₹)", 0, int(income//2), 0, 500, key=f"sim{i}")
                    if extra>0:
                        nm = remaining/(cf+extra) if (cf+extra)>0 else 9999
                        nd = datetime.now()+relativedelta(months=int(nm))
                        st.success(f"💡 +₹{extra:,.0f}/month → **{nd.strftime('%B %Y')}** — {max(0,real_m-nm):.0f} months earlier!")
                with c2:
                    fig,ax = plt.subplots(figsize=(3,3),facecolor='#0b0f1e')
                    ax.set_facecolor('#0b0f1e')
                    ax.pie([prog,100-prog],colors=[bc,'#131720'],startangle=90,
                           wedgeprops=dict(width=0.4,edgecolor='#0b0f1e'))
                    ax.text(0,0,f"{prog:.0f}%",ha='center',va='center',fontsize=18,fontweight='bold',color=bc)
                    plt.tight_layout(); st.pyplot(fig); plt.close()

                with st.expander(f"📅 Month-by-Month Plan — {goal['name']}"):
                    plan=[]; running=savings
                    for m in range(1,min(int(goal['months'])+3,25)):
                        ms=min(npm,cf); running+=ms; pct=min(100,running/goal['cost']*100)
                        plan.append({'Month':f'Month {m}','Save':f'₹{ms:,.0f}',
                                     'Cumulative':f'₹{running:,.0f}','Progress':f'{pct:.1f}%',
                                     'Status':'✅ Done' if running>=goal['cost'] else '🔄 In progress'})
                        if running>=goal['cost']: break
                    st.dataframe(pd.DataFrame(plan),use_container_width=True,hide_index=True)

        # ══ BUDGET ══════════════════════════════
        elif active == "budget":
            page_header("📊","Budget Analyser","50/30/20 framework · gap analysis","#4f8ef7")
            ni=income*0.50; wi=income*0.30; si=income*0.20
            eg=expenses-ni; sg=si-savings
            ideal_s = float(np.clip((lr_model.predict([[income,ni,si,model_debt]])[0]+
                                     xgb_model.predict([[income,ni,si,model_debt]])[0])/2,0,100))

            components.html(f"""
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Syne:wght@800&family=Space+Grotesk:wght@400;600;700&display=swap');
            *{{font-family:'Space Grotesk',sans-serif;box-sizing:border-box;}}
            .bc{{background:#0b0f1e;border-radius:20px;padding:30px;border:1px solid #1a2040;margin:0 48px 16px;}}
            .bct{{font-size:10px;color:#4f8ef7;text-transform:uppercase;letter-spacing:3px;font-weight:600;margin-bottom:22px;}}
            .brow{{display:flex;justify-content:space-between;align-items:center;padding:16px 0;border-bottom:1px solid #0f1528;}}
            .brow:last-child{{border:none;}}
            .bleft{{display:flex;align-items:center;gap:14px;}}
            .bic{{width:38px;height:38px;border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:17px;}}
            .blbl{{font-size:15px;font-weight:600;color:#fff;}}
            .bsub{{font-size:11px;color:#3a4560;margin-top:2px;}}
            .bright{{text-align:right;}}
            .bidl{{font-family:'Syne',sans-serif;font-size:21px;font-weight:700;}}
            .byrs{{font-size:12px;margin-top:3px;}}
            .bbar{{background:#0f1528;border-radius:999px;height:4px;margin-top:8px;overflow:hidden;}}
            .bfill{{height:100%;border-radius:999px;}}
            </style>
            <div class="bc">
                <div class="bct">Recommended split of ₹{income:,.0f}/month</div>
                <div class="brow">
                    <div class="bleft"><div class="bic" style="background:#0a1e3d;">🏠</div>
                        <div><div class="blbl">Needs</div><div class="bsub">Rent, food, bills, transport</div></div></div>
                    <div class="bright">
                        <div class="bidl" style="color:#4f8ef7">₹{ni:,.0f}</div>
                        <div class="byrs" style="color:{'#00e5a0' if expenses<=ni else '#f74f4f'}">You spend ₹{expenses:,.0f}</div>
                        <div class="bbar"><div class="bfill" style="width:{min(100,expenses/ni*100):.0f}%;background:{'#00e5a0' if expenses<=ni else '#f74f4f'};"></div></div>
                    </div>
                </div>
                <div class="brow">
                    <div class="bleft"><div class="bic" style="background:#2a1a00;">🎮</div>
                        <div><div class="blbl">Wants</div><div class="bsub">Entertainment, dining, hobbies</div></div></div>
                    <div class="bright"><div class="bidl" style="color:#f7c94f">₹{wi:,.0f}</div><div class="byrs" style="color:#3a4560">Guideline</div></div>
                </div>
                <div class="brow">
                    <div class="bleft"><div class="bic" style="background:#09200f;">🏦</div>
                        <div><div class="blbl">Savings + Goals</div><div class="bsub">Investments, emergency, goals</div></div></div>
                    <div class="bright">
                        <div class="bidl" style="color:#00e5a0">₹{si:,.0f}</div>
                        <div class="byrs" style="color:{'#00e5a0' if savings>=si else '#f74f4f'}">You save ₹{savings:,.0f}</div>
                        <div class="bbar"><div class="bfill" style="width:{min(100,savings/si*100):.0f}%;background:{'#00e5a0' if savings>=si else '#f74f4f'};"></div></div>
                    </div>
                </div>
            </div>
            """, height=340)

            c1,c2 = st.columns(2)
            with c1:
                st.markdown("**📋 Gap Analysis**")
                if eg>0: st.error(f"❌ Expenses exceed 50% by ₹{eg:,.0f}/month")
                else:    st.success(f"✅ Expenses ₹{abs(eg):,.0f} under 50% limit")
                if sg>0: st.warning(f"⚠️ Save ₹{sg:,.0f} more to hit 20% target")
                else:    st.success(f"✅ Savings ₹{abs(sg):,.0f} above 20% target")
                if ideal_s>final_score: st.info(f"💡 Following 50/30/20 could boost score by **{ideal_s-final_score:.1f} pts** → {ideal_s:.0f}/100")

                st.markdown("**🎚️ Budget Optimizer**")
                ne = st.slider("Reduce expenses by (₹)", 0, int(expenses*0.5), 0, 1000)
                ns = st.slider("Increase savings by (₹)", 0, int(cf if cf>0 else 1), 0, 500)
                if ne>0 or ns>0:
                    ae=expenses-ne; as_=savings+ns
                    ns_=float(np.clip((lr_model.predict([[income,ae,as_,model_debt]])[0]+
                                       xgb_model.predict([[income,ae,as_,model_debt]])[0])/2,0,100))
                    st.success(f"📈 Adjusted score: **{ns_:.0f}/100** (+{ns_-final_score:.1f} pts)")

            with c2:
                fig,ax = plt.subplots(figsize=(5,4),facecolor='#0b0f1e')
                dark_chart(ax,fig)
                cats=['Needs\n50%','Wants\n30%','Save\n20%','Your\nExpenses','Your\nSavings']
                vals=[ni,wi,si,expenses,savings]
                clrs=['#1a4fa8','#7a4a00','#1a6b3a','#8b1a1a' if expenses>ni else '#1a6b3a','#1a6b3a']
                bars=ax.bar(cats,vals,color=clrs,edgecolor='#0b0f1e',width=0.5,zorder=3)
                for b,v in zip(bars,vals):
                    ax.text(b.get_x()+b.get_width()/2,b.get_height()+income*0.01,
                            f'₹{v/1000:.0f}k',ha='center',color='#3a4560',fontsize=9)
                ax.set_axisbelow(True); ax.yaxis.grid(True,color='#0f1528',linewidth=0.8)
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f'₹{x/1000:.0f}k'))
                plt.tight_layout(); st.pyplot(fig); plt.close()

        # ══ METRICS ═════════════════════════════
        elif active == "metrics":
            page_header("📈","Key Metrics","financial ratios · health indicators","#00e5a0")
            components.html(f"""
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Syne:wght@800&family=Space+Grotesk:wght@400;600;700&display=swap');
            *{{font-family:'Space Grotesk',sans-serif;box-sizing:border-box;}}
            .mg{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;padding:0 48px 16px;}}
            .mc{{background:#0b0f1e;border-radius:18px;padding:26px;border:1px solid #1a2040;
                  animation:si 0.4s ease both;}}
            .mc:nth-child(2){{animation-delay:0.07s;}}
            .mc:nth-child(3){{animation-delay:0.14s;}}
            .mc:nth-child(4){{animation-delay:0.21s;}}
            .mc:nth-child(5){{animation-delay:0.28s;}}
            .mc:nth-child(6){{animation-delay:0.35s;}}
            @keyframes si{{from{{opacity:0;transform:translateY(18px)}}to{{opacity:1;transform:translateY(0)}}}}
            .mi{{font-size:26px;margin-bottom:12px;}}
            .mv{{font-family:'Syne',sans-serif;font-size:34px;font-weight:800;line-height:1;}}
            .ml{{font-size:10px;color:#3a4560;text-transform:uppercase;letter-spacing:2px;margin-top:9px;}}
            .mid{{font-size:11px;margin-top:5px;}}
            .mb{{background:#0f1528;border-radius:999px;height:4px;margin-top:12px;overflow:hidden;}}
            .mf{{height:100%;border-radius:999px;}}
            .mv2{{display:inline-block;margin-top:11px;padding:3px 11px;border-radius:99px;font-size:10px;font-weight:700;}}
            </style>
            <div class="mg">
                <div class="mc">
                    <div class="mi">💸</div>
                    <div class="mv" style="color:{'#00e5a0' if er<=70 else '#f7c94f' if er<=90 else '#f74f4f'}">{er:.1f}%</div>
                    <div class="ml">Expense Ratio</div>
                    <div class="mid" style="color:#3a4560">Ideal: under 70%</div>
                    <div class="mb"><div class="mf" style="width:{min(100,er):.0f}%;background:{'#00e5a0' if er<=70 else '#f7c94f' if er<=90 else '#f74f4f'};"></div></div>
                    <span class="mv2" style="background:{'#091f14' if er<=70 else '#1f1500' if er<=90 else '#1f0909'};color:{'#00e5a0' if er<=70 else '#f7c94f' if er<=90 else '#f74f4f'}">
                        {'✅ Healthy' if er<=70 else '⚠️ Moderate' if er<=90 else '❌ High'}
                    </span>
                </div>
                <div class="mc">
                    <div class="mi">🏦</div>
                    <div class="mv" style="color:{'#00e5a0' if sr>=20 else '#f7c94f' if sr>=10 else '#f74f4f'}">{sr:.1f}%</div>
                    <div class="ml">Savings Rate</div>
                    <div class="mid" style="color:#3a4560">Ideal: above 20%</div>
                    <div class="mb"><div class="mf" style="width:{min(100,sr):.0f}%;background:{'#00e5a0' if sr>=20 else '#f7c94f' if sr>=10 else '#f74f4f'};"></div></div>
                    <span class="mv2" style="background:{'#091f14' if sr>=20 else '#1f1500' if sr>=10 else '#1f0909'};color:{'#00e5a0' if sr>=20 else '#f7c94f' if sr>=10 else '#f74f4f'}">
                        {'✅ Excellent' if sr>=20 else '⚠️ Low' if sr>=10 else '❌ Very Low'}
                    </span>
                </div>
                <div class="mc">
                    <div class="mi">💳</div>
                    <div class="mv" style="color:{'#00e5a0' if dti_pct<=15 else '#f7c94f' if dti_pct<=36 else '#f74f4f'}">{dti_pct:.1f}%</div>
                    <div class="ml">Debt-to-Income</div>
                    <div class="mid" style="color:#3a4560">Ideal: under 36%</div>
                    <div class="mb"><div class="mf" style="width:{min(100,dti_pct):.0f}%;background:{'#00e5a0' if dti_pct<=15 else '#f7c94f' if dti_pct<=36 else '#f74f4f'};"></div></div>
                    <span class="mv2" style="background:{'#091f14' if dti_pct<=15 else '#1f1500' if dti_pct<=36 else '#1f0909'};color:{'#00e5a0' if dti_pct<=15 else '#f7c94f' if dti_pct<=36 else '#f74f4f'}">
                        {'✅ Excellent' if dti_pct<=15 else '⚠️ Moderate' if dti_pct<=36 else '❌ Dangerous'}
                    </span>
                </div>
                <div class="mc">
                    <div class="mi">💵</div>
                    <div class="mv" style="color:{'#00e5a0' if cf>=0 else '#f74f4f'}">₹{cf:,.0f}</div>
                    <div class="ml">Monthly Surplus</div>
                    <div class="mid" style="color:#3a4560">Income minus expenses</div>
                    <div class="mb"><div class="mf" style="width:{'55' if cf>=0 else '100'}%;background:{'#00e5a0' if cf>=0 else '#f74f4f'};"></div></div>
                    <span class="mv2" style="background:{'#091f14' if cf>=0 else '#1f0909'};color:{'#00e5a0' if cf>=0 else '#f74f4f'}">{'✅ Positive' if cf>=0 else '❌ Deficit'}</span>
                </div>
                <div class="mc">
                    <div class="mi">📅</div>
                    <div class="mv" style="color:#4f8ef7">{income/expenses:.2f}x</div>
                    <div class="ml">Income/Expense Ratio</div>
                    <div class="mid" style="color:#3a4560">Ideal: above 1.5x</div>
                    <div class="mb"><div class="mf" style="width:{min(100,(income/expenses if expenses>0 else 2)/2*100):.0f}%;background:#4f8ef7;"></div></div>
                    <span class="mv2" style="background:#0a1535;color:#4f8ef7">{'✅ Strong' if income/expenses>=1.5 else '⚠️ Tight' if income/expenses>=1.1 else '❌ Weak'}</span>
                </div>
                <div class="mc">
                    <div class="mi">🎯</div>
                    <div class="mv" style="color:{hex_color}">{final_score:.0f}</div>
                    <div class="ml">Health Score</div>
                    <div class="mid" style="color:#3a4560">ML composite score</div>
                    <div class="mb"><div class="mf" style="width:{final_score:.0f}%;background:{hex_color};"></div></div>
                    <span class="mv2" style="background:{hex_color}18;color:{hex_color}">{label}</span>
                </div>
            </div>
            """, height=500)

            st.markdown("**📉 Historical Health Score Trend**")
            fig,ax = plt.subplots(figsize=(10,3.5),facecolor='#0b0f1e')
            dark_chart(ax,fig)
            ax.plot(monthly['Month'],monthly['health_score'],color='#4f8ef7',linewidth=2.5,marker='o',markersize=4)
            ax.fill_between(monthly['Month'],monthly['health_score'],alpha=0.07,color='#4f8ef7')
            ax.axhline(75,color='#00e5a0',linewidth=1,linestyle='--',alpha=0.5,label='Excellent')
            ax.axhline(50,color='#f7c94f',linewidth=1,linestyle='--',alpha=0.5,label='Good')
            ax.legend(facecolor='#0b0f1e',labelcolor='#3a4560',fontsize=9,framealpha=0.4)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        # ══ SPENDING ════════════════════════════
        elif active == "spending":
            page_header("🛒","Spending Analysis","category breakdown · top expenses","#f7c94f")
            cat_df  = df[df['Transaction Type']=='debit'].groupby('Category')['Amount'].sum().sort_values(ascending=False)
            total_s = cat_df.sum()
            c1,c2   = st.columns([1,1])
            with c1:
                cd=cat_df.head(10).reset_index(); cd.columns=['Category','Amount']
                cd['% Total']=(cd['Amount']/total_s*100).apply(lambda x:f"{x:.1f}%")
                cd['Amount']=cd['Amount'].apply(lambda x:f'₹{x:,.0f}')
                st.dataframe(cd,use_container_width=True,hide_index=True)
                st.markdown(f"**📅 Monthly Trend — {top_3_cats[0]}**")
                cm=df[(df['Transaction Type']=='debit')&(df['Category']==top_3_cats[0])].groupby('Month')['Amount'].sum()
                fig2,ax2=plt.subplots(figsize=(5,2.5),facecolor='#0b0f1e'); dark_chart(ax2,fig2)
                ax2.bar(range(len(cm)),cm.values,color='#f7c94f',edgecolor='#0b0f1e',alpha=0.9)
                ax2.set_xticks(range(len(cm))); ax2.set_xticklabels([d.strftime('%b') for d in cm.index],fontsize=7,rotation=45)
                ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f'₹{x/1000:.0f}k'))
                plt.tight_layout(); st.pyplot(fig2); plt.close()
            with c2:
                fig,ax=plt.subplots(figsize=(5,5),facecolor='#0b0f1e'); ax.set_facecolor('#0b0f1e')
                top8=cat_df.head(8); clrs=['#4f8ef7','#00e5a0','#f7c94f','#f74f4f','#a78bfa','#e879f9','#38bdf8','#fb923c']
                ws,ts,ats=ax.pie(top8.values,labels=top8.index,colors=clrs[:len(top8)],
                                  autopct='%1.0f%%',startangle=90,
                                  wedgeprops=dict(width=0.55,edgecolor='#0b0f1e'),pctdistance=0.8)
                for t in ts: t.set_color('#3a4560'); t.set_fontsize(8)
                for a in ats: a.set_color('#fff'); a.set_fontsize(8); a.set_fontweight('bold')
                ax.text(0,0,f'₹{total_s/1000:.0f}k\ntotal',ha='center',va='center',fontsize=11,color='#fff',fontweight='bold')
                plt.tight_layout(); st.pyplot(fig); plt.close()
                st.markdown("**📈 Income vs Expenses Trend**")
                fig3,ax3=plt.subplots(figsize=(5,2.5),facecolor='#0b0f1e'); dark_chart(ax3,fig3)
                ax3.plot(monthly['Month'],monthly['income'],color='#00e5a0',linewidth=2,label='Income')
                ax3.plot(monthly['Month'],monthly['expenses'],color='#f74f4f',linewidth=2,label='Expenses')
                ax3.fill_between(monthly['Month'],monthly['income'],monthly['expenses'],
                                  where=monthly['income']>monthly['expenses'],alpha=0.07,color='#00e5a0')
                ax3.legend(facecolor='#0b0f1e',labelcolor='#888',fontsize=8,framealpha=0.4)
                ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f'₹{x/1000:.0f}k'))
                plt.tight_layout(); st.pyplot(fig3); plt.close()

        # ══ EMERGENCY FUND ══════════════════════
        elif active == "emergency":
            page_header("🛡️","Emergency Fund","your financial safety net","#f74f4f")
            ef_shortfall = max(0,ef_target-emergency_savings)
            ef_months_needed = ef_shortfall/savings if savings>0 else 9999
            ef_date = datetime.now()+relativedelta(months=int(min(ef_months_needed,999)))
            ef_col  = '#00e5a0' if ef_pct>=100 else '#f7c94f' if ef_pct>=50 else '#f74f4f'

            components.html(f"""
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Syne:wght@800&family=Space+Grotesk:wght@400;600;700&display=swap');
            *{{font-family:'Space Grotesk',sans-serif;box-sizing:border-box;}}
            .ec{{background:#0b0f1e;border-radius:20px;padding:30px;border:1px solid #1a2040;margin:0 48px 16px;}}
            .ep{{font-family:'Syne',sans-serif;font-size:58px;font-weight:800;color:{ef_col};line-height:1;}}
            .el{{font-size:10px;color:#3a4560;text-transform:uppercase;letter-spacing:2px;margin-top:6px;}}
            .et{{background:#0f1528;border-radius:999px;height:12px;margin:16px 0 8px;overflow:hidden;}}
            .ef{{height:100%;border-radius:999px;
                  background:{'linear-gradient(90deg,#00a070,#00e5a0)' if ef_pct>=100 else 'linear-gradient(90deg,#f74f4f,#f7c94f)'};
                  width:{ef_pct:.0f}%;box-shadow:0 0 20px {'#00e5a055' if ef_pct>=100 else '#f74f4f55'};}}
            .em{{display:flex;justify-content:space-between;font-size:10px;color:#3a4560;}}
            .ec2{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:18px;}}
            .ecc{{background:#0f1528;border-radius:12px;padding:16px;text-align:center;}}
            .ecv{{font-family:'Syne',sans-serif;font-size:19px;font-weight:700;color:#fff;}}
            .ecl{{font-size:9px;color:#3a4560;text-transform:uppercase;letter-spacing:1px;margin-top:4px;}}
            .er{{margin-top:18px;padding:14px;border-radius:12px;
                  background:{'#091f14' if ef_pct>=100 else '#1f0a09'};
                  border:1px solid {'#00e5a025' if ef_pct>=100 else '#f74f4f25'};
                  font-size:13px;color:{'#00e5a0' if ef_pct>=100 else '#f74f4f'};font-weight:600;}}
            </style>
            <div class="ec">
                <div class="ep">{ef_pct:.0f}%</div>
                <div class="el">Emergency Fund Funded</div>
                <div class="et"><div class="ef"></div></div>
                <div class="em"><span>₹{emergency_savings:,.0f} saved</span><span>₹{ef_target:,.0f} target ({emergency_months}× expenses)</span></div>
                <div class="ec2">
                    <div class="ecc"><div class="ecv">₹{ef_target:,.0f}</div><div class="ecl">Target</div></div>
                    <div class="ecc"><div class="ecv">₹{ef_shortfall:,.0f}</div><div class="ecl">Shortfall</div></div>
                    <div class="ecc"><div class="ecv">{ef_date.strftime('%b %Y') if ef_pct<100 else 'Done!'}</div><div class="ecl">Est. Complete</div></div>
                </div>
                <div class="er">
                    {'✅ Fully funded! Your emergency fund covers ' + str(emergency_months) + ' months of expenses.' if ef_pct>=100
                     else '⚠️ At current savings rate, funded by ' + ef_date.strftime('%B %Y') + '.'}
                </div>
            </div>
            """, height=370)

            c1,c2 = st.columns(2)
            with c1:
                st.markdown("**💡 Emergency Fund Booster**")
                extra_ef = st.slider("Extra monthly allocation (₹)", 0, int(max(cf,1)), 0, 500, key="efb")
                if extra_ef>0:
                    nm=ef_shortfall/(savings+extra_ef) if (savings+extra_ef)>0 else 9999
                    nd=datetime.now()+relativedelta(months=int(nm))
                    st.success(f"💡 +₹{extra_ef:,.0f}/month → Funded by **{nd.strftime('%B %Y')}**")

                components.html("""
                <style>
                @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500&display=swap');
                *{{font-family:'Space Grotesk',sans-serif;box-sizing:border-box;}}
                .rl{{background:#0b0f1e;border-radius:14px;padding:18px;border:1px solid #1a2040;margin-top:12px;}}
                .ri{{display:flex;gap:12px;padding:10px 0;border-bottom:1px solid #0f1528;font-size:12px;color:#888;}}
                .ri:last-child{{border:none;}}
                .rn{{color:#f74f4f;font-weight:700;min-width:28px;}}
                </style>
                <div class="rl">
                    <div style="font-size:10px;color:#f74f4f;text-transform:uppercase;letter-spacing:2px;font-weight:600;margin-bottom:10px;">Industry Rules</div>
                    <div class="ri"><span class="rn">3×</span><span>Minimum — freelancers & students</span></div>
                    <div class="ri"><span class="rn">6×</span><span>Recommended — salaried employees</span></div>
                    <div class="ri"><span class="rn">12×</span><span>Conservative — self-employed / business</span></div>
                    <div class="ri"><span class="rn">✓</span><span>Keep in high-yield liquid savings account</span></div>
                    <div class="ri"><span class="rn">✗</span><span>Never invest it in stocks or crypto</span></div>
                </div>
                """, height=210)

            with c2:
                fig,ax=plt.subplots(figsize=(4.5,4.5),facecolor='#0b0f1e'); ax.set_facecolor('#0b0f1e')
                theta=np.linspace(0,np.pi,200)
                ax.plot(np.cos(theta),np.sin(theta),color='#0f1528',linewidth=22,solid_capstyle='round')
                ft=np.linspace(0,np.pi*min(ef_pct,100)/100,200)
                ax.plot(np.cos(ft),np.sin(ft),color=ef_col,linewidth=22,solid_capstyle='round')
                ax.text(0,0.2,f"{ef_pct:.0f}%",ha='center',va='center',fontsize=30,fontweight='bold',color=ef_col)
                ax.text(0,-0.1,"Funded",ha='center',fontsize=12,color='#3a4560')
                ax.set_xlim(-1.4,1.4); ax.set_ylim(-0.5,1.4); ax.axis('off')
                plt.tight_layout(); st.pyplot(fig); plt.close()

        # ══ PROJECTION ══════════════════════════
        elif active == "projection":
            page_header("🔮","6-Month Projection","scenario planning · savings forecast","#38bdf8")
            c1,c2,c3 = st.columns(3)
            with c1: ec=st.slider("Expense change (%/month)",-30,30,-2,1)
            with c2: sc=st.slider("Savings change (%/month)",-20,50,3,1)
            with c3: ic=st.slider("Income growth (%/month)",0,10,0,1)

            rows=[]; rs=savings
            for m in range(1,7):
                pi=income*(1+ic/100)**m; pe=expenses*(1+ec/100)**m; ps=savings*(1+sc/100)**m
                rs+=ps
                pl=float(np.clip((lr_model.predict([[pi,pe,rs,model_debt]])[0]+
                                   xgb_model.predict([[pi,pe,rs,model_debt]])[0])/2,0,100))
                rows.append({'Month':(datetime.now()+relativedelta(months=m)).strftime('%b %Y'),
                             'Income':pi,'Expenses':pe,'Surplus':pi-pe,'Savings':rs,'Score':pl})
            pdf=pd.DataFrame(rows)

            c1,c2=st.columns(2)
            with c1:
                dd=pdf.copy()
                for col in ['Income','Expenses','Surplus','Savings']:
                    dd[col]=dd[col].apply(lambda x:f'₹{x:,.0f}')
                dd['Score']=dd['Score'].apply(lambda x:f'{x:.0f}/100')
                st.dataframe(dd,use_container_width=True,hide_index=True)
                delta=pdf['Score'].iloc[-1]-final_score
                if delta>0:   st.success(f"📈 Score improves **+{delta:.1f} pts** → {pdf['Score'].iloc[-1]:.0f}/100")
                elif delta<0: st.warning(f"📉 Score drops **{delta:.1f} pts** → {pdf['Score'].iloc[-1]:.0f}/100")
                else:         st.info("📊 Score stays stable")

                # Monthly savings built up
                st.markdown("**📊 Cumulative Savings Built**")
                st.metric("After 6 months", f"₹{pdf['Savings'].iloc[-1]:,.0f}",
                          delta=f"+₹{pdf['Savings'].iloc[-1]-savings:,.0f}")

            with c2:
                fig,(ax1,ax2)=plt.subplots(2,1,figsize=(5,6),facecolor='#0b0f1e')
                for ax in [ax1,ax2]: dark_chart(ax,fig)
                mx=pdf['Month']
                ax1.plot(mx,pdf['Income'],color='#00e5a0',linewidth=2.5,marker='o',markersize=5,label='Income')
                ax1.plot(mx,pdf['Expenses'],color='#f74f4f',linewidth=2.5,marker='s',markersize=5,label='Expenses')
                ax1.fill_between(mx,pdf['Income'],pdf['Expenses'],
                                  where=pdf['Income']>pdf['Expenses'],alpha=0.06,color='#00e5a0')
                ax1.legend(facecolor='#0b0f1e',labelcolor='#888',fontsize=8); ax1.set_title('Income vs Expenses',color='#3a4560',fontsize=10)
                ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f'₹{x/1000:.0f}k')); ax1.tick_params(axis='x',rotation=30)
                ax2.plot(mx,pdf['Score'],color='#4f8ef7',linewidth=2.5,marker='D',markersize=5)
                ax2.fill_between(mx,pdf['Score'],alpha=0.06,color='#4f8ef7')
                ax2.axhline(final_score,color='#f7c94f',linewidth=1,linestyle='--',alpha=0.6,label='Current')
                ax2.set_title('Projected Health Score',color='#3a4560',fontsize=10)
                ax2.legend(facecolor='#0b0f1e',labelcolor='#888',fontsize=8); ax2.tick_params(axis='x',rotation=30)
                plt.tight_layout(); st.pyplot(fig); plt.close()

        # ══ NET WORTH ═══════════════════════════
        elif active == "networth":
            page_header("💎","Net Worth Tracker","assets · liabilities · wealth snapshot","#e879f9")
            c1,c2=st.columns(2)
            with c1:
                st.markdown("**📦 Assets**")
                bb = st.number_input("Bank Balance (₹)",          min_value=0.0,value=float(savings*6),step=1000.0)
                iv = st.number_input("Investments / Stocks (₹)",  min_value=0.0,value=50000.0,step=1000.0)
                re = st.number_input("Property Value (₹)",         min_value=0.0,value=0.0,step=10000.0)
                oa = st.number_input("Other Assets (₹)",          min_value=0.0,value=0.0,step=1000.0)
                ta = bb+iv+re+oa
            with c2:
                st.markdown("**📤 Liabilities**")
                hl = st.number_input("Home Loan Outstanding (₹)",  min_value=0.0,value=0.0,step=10000.0)
                cl = st.number_input("Car Loan Outstanding (₹)",   min_value=0.0,value=0.0,step=5000.0)
                cc = st.number_input("Credit Card Debt (₹)",       min_value=0.0,value=float(debt*12),step=1000.0)
                od = st.number_input("Other Liabilities (₹)",      min_value=0.0,value=0.0,step=1000.0)
                tl = hl+cl+cc+od

            nw=ta-tl; nc="#00e5a0" if nw>=0 else "#f74f4f"
            components.html(f"""
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Syne:wght@800&family=Space+Grotesk:wght@400;600;700&display=swap');
            *{{font-family:'Space Grotesk',sans-serif;box-sizing:border-box;}}
            .nw{{background:#0b0f1e;border-radius:20px;padding:28px;border:1px solid #1a2040;
                  margin:16px 0;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:20px;}}
            .nl{{font-size:10px;color:#3a4560;text-transform:uppercase;letter-spacing:3px;margin-bottom:6px;}}
            .nv{{font-family:'Syne',sans-serif;font-size:48px;font-weight:800;color:{nc};line-height:1;}}
            .ns{{font-size:13px;color:#3a4560;margin-top:8px;}}
            .nr{{display:flex;gap:20px;}}
            .nrc{{background:#0f1528;border-radius:14px;padding:18px 24px;text-align:center;}}
            .ncv{{font-family:'Syne',sans-serif;font-size:20px;font-weight:700;}}
            .ncl{{font-size:9px;color:#3a4560;text-transform:uppercase;letter-spacing:1px;margin-top:5px;}}
            </style>
            <div class="nw">
                <div><div class="nl">Your Net Worth</div>
                    <div class="nv">₹{abs(nw):,.0f}</div>
                    <div class="ns">{'🎉 Positive net worth' if nw>=0 else '⚠️ Negative — focus on debt payoff'}</div>
                </div>
                <div class="nr">
                    <div class="nrc"><div class="ncv" style="color:#00e5a0">₹{ta:,.0f}</div><div class="ncl">Total Assets</div></div>
                    <div class="nrc"><div class="ncv" style="color:#f74f4f">₹{tl:,.0f}</div><div class="ncl">Total Liabilities</div></div>
                </div>
            </div>
            """, height=180)

            c1,c2=st.columns(2)
            with c1:
                if ta>0:
                    fig,ax=plt.subplots(figsize=(4.5,3.5),facecolor='#0b0f1e'); ax.set_facecolor('#0b0f1e')
                    al=[('Bank',bb),('Investments',iv),('Property',re),('Other',oa)]
                    al=[(l,v) for l,v in al if v>0]
                    if al:
                        ls,vs=zip(*al)
                        ax.pie(vs,labels=ls,colors=['#4f8ef7','#00e5a0','#a78bfa','#38bdf8'][:len(vs)],
                               autopct='%1.0f%%',startangle=90,wedgeprops=dict(width=0.5,edgecolor='#0b0f1e'))
                        ax.set_title('Assets Breakdown',color='#3a4560',fontsize=10,pad=10)
                    plt.tight_layout(); st.pyplot(fig); plt.close()

                # Debt payoff timeline
                st.markdown("**💳 Debt Payoff Timeline**")
                if tl>0 and cf>0:
                    months_to_payoff = tl/cf
                    payoff_date = datetime.now()+relativedelta(months=int(months_to_payoff))
                    st.info(f"At current surplus (₹{cf:,.0f}/month), debt-free by **{payoff_date.strftime('%B %Y')}** (~{months_to_payoff:.0f} months)")
                elif tl==0:
                    st.success("✅ Debt free!")

            with c2:
                st.markdown("**📈 Net Worth Growth (12 months)**")
                nwp=[nw+cf*m for m in range(13)]
                fig2,ax2=plt.subplots(figsize=(4.5,3.5),facecolor='#0b0f1e'); dark_chart(ax2,fig2)
                ax2.plot(range(13),nwp,color='#e879f9',linewidth=2.5,marker='o',markersize=4)
                ax2.fill_between(range(13),nwp,alpha=0.06,color='#e879f9')
                ax2.axhline(0,color='#3a4560',linewidth=0.8,linestyle='--')
                ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f'₹{x/1000:.0f}k'))
                ax2.set_xlabel('Month',color='#3a4560')
                plt.tight_layout(); st.pyplot(fig2); plt.close()
                st.metric("Net Worth in 12 months",f"₹{nwp[-1]:,.0f}",delta=f"+₹{nwp[-1]-nw:,.0f}")

        # ══ AI ADVISOR ══════════════════════════
        elif active == "ai":
            page_header("🤖","AI Financial Advisor",f"personalised advice for {name}","#00d4ff")

            # Auto-load initial tips
            if len(st.session_state.chat_history)==0:
                with st.spinner("Loading your personalised tips..."):
                    try:
                        gt="\n".join([f"- {g['goal']['name']}: ₹{g['goal']['cost']:,.0f} in {g['goal']['months']:.0f} months, "
                                       f"{'achievable' if g['achievable'] else 'needs more savings'}" for g in goal_results])
                        r=client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[{"role":"user","content":
                                f"Give {name} exactly 5 short, friendly, specific financial tips to improve their score from {final_score:.0f}/100.\n"
                                f"Each tip should be 1-2 sentences. Use ₹ amounts. Be actionable.\n\n"
                                f"Income: ₹{income:,.0f} | Expenses: ₹{expenses:,.0f} | Savings: ₹{savings:,.0f} | Surplus: ₹{cf:,.0f}\n"
                                f"Goals: {gt}\nTop spending: {', '.join(top_3_cats[:3])}"}],
                            max_tokens=400)
                        st.session_state.chat_history=[{"role":"assistant","content":r.choices[0].message.content}]
                        st.rerun()
                    except Exception as e:
                        st.warning(f"AI unavailable: {e}")

            # Display chat
            for msg in st.session_state.chat_history:
                if msg['role']=='user':
                    st.markdown(f"""<div style="background:#0a1535;border:1px solid #1a2040;border-radius:16px 16px 4px 16px;
                        padding:14px 18px;margin:6px 80px 6px 0;font-size:13px;color:#fff;">👤 {msg['content']}</div>""",
                        unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div style="background:#091820;border:1px solid #00d4ff22;border-radius:16px 16px 16px 4px;
                        padding:14px 18px;margin:6px 0 6px 80px;font-size:13px;color:#cde;line-height:1.7;">🤖 {msg['content']}</div>""",
                        unsafe_allow_html=True)

            # Quick question chips
            st.markdown("**💬 Quick questions:**")
            qcols=st.columns(5)
            quick_qs=["Analyse my spending","How to reach goals faster?","Pay debt or save first?","Build emergency fund","Investment ideas for surplus"]
            for col,q in zip(qcols,quick_qs):
                with col:
                    if st.button(q, key=f"qq_{q[:8]}"):
                        st.session_state.chat_history.append({"role":"user","content":q})
                        st.rerun()

            c1,c2=st.columns([5,1])
            with c1: user_q=st.text_input("Ask anything...",placeholder="e.g. How can I save faster for my car?",key="ai_in",label_visibility="collapsed")
            with c2: send=st.button("Send 💬")

            if send and user_q.strip():
                st.session_state.chat_history.append({"role":"user","content":user_q.strip()})
                st.rerun()

            # Process last unanswered user message
            if st.session_state.chat_history and st.session_state.chat_history[-1]['role']=='user':
                with st.spinner("Thinking..."):
                    try:
                        gt="\n".join([f"- {g['goal']['name']}: ₹{g['goal']['cost']:,.0f}, {'achievable' if g['achievable'] else 'needs more time'}" for g in goal_results])
                        sys=f"""You are a friendly personal financial advisor for {name} (age {age}, {occ}).
Income: ₹{income:,.0f} | Expenses: ₹{expenses:,.0f} | Savings: ₹{savings:,.0f} | Debt EMI: ₹{debt:,.0f} | Surplus: ₹{cf:,.0f}
Score: {final_score:.0f}/100 ({label}) | Emergency Fund: {ef_pct:.0f}% funded
Goals: {gt}
Top spending: {', '.join(top_3_cats[:3])}
Be specific with ₹ amounts. Friendly tone. Under 200 words. Actionable."""
                        msgs=[{"role":"system","content":sys}]+[{"role":h['role'],"content":h['content']} for h in st.session_state.chat_history]
                        r=client.chat.completions.create(model="llama-3.3-70b-versatile",messages=msgs,max_tokens=400)
                        st.session_state.chat_history.append({"role":"assistant","content":r.choices[0].message.content})
                        st.rerun()
                    except Exception as e:
                        st.error(f"AI unavailable: {e}")

            if st.button("🗑️ Clear Chat",key="clr"):
                st.session_state.chat_history=[]; st.rerun()

        back_button()
        bottom_nav(active)

        st.markdown("<br>", unsafe_allow_html=True)
        c1,c2,c3=st.columns([3,1,3])
        with c2:
            if st.button("↺ Start Over",key="rst_i"):
                st.session_state.screen=1; st.session_state.active_page=None
                st.session_state.chat_history=[]; st.rerun()
