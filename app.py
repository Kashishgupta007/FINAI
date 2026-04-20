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
[data-testid="stAppViewContainer"] { background-color: #0f1117; }
[data-testid="stSidebar"] { background-color: #0f1117; }
* { font-family: 'Segoe UI', sans-serif; }
.big-title { font-size: 38px; font-weight: 800; color: #fff; text-align: center; padding: 30px 0 5px; }
.small-sub { font-size: 15px; color: #555; text-align: center; margin-bottom: 40px; }
.sec { font-size: 18px; font-weight: 700; color: #fff; margin: 32px 0 12px; padding-left: 12px; border-left: 3px solid #378ADD; }
.card { background: #16181f; border-radius: 12px; padding: 18px; border: 1px solid #1e2030; text-align: center; }
.card-val { font-size: 26px; font-weight: 700; color: #fff; }
.card-lbl { font-size: 12px; color: #555; margin-top: 4px; }
.score-wrap { background: #16181f; border-radius: 16px; padding: 32px; text-align: center; border: 1px solid #1e2030; }
.score-num { font-size: 80px; font-weight: 900; line-height: 1; }
.score-lbl { font-size: 20px; font-weight: 600; margin-top: 6px; }
.goal-wrap { background: #16181f; border-radius: 12px; padding: 20px; border: 1px solid #1e2030; margin-bottom: 20px; }
div[data-testid="stNumberInput"] label { color: #aaa !important; font-size: 13px !important; }
div[data-testid="stTextInput"] label { color: #aaa !important; font-size: 13px !important; }
</style>
""", unsafe_allow_html=True)

# ── Load data (cached) ────────────────────────────
@st.cache_data
def load_default_data():
    return pd.read_excel("personal_transactions_dashboard_ready (2).xlsx")

@st.cache_data
def load_uploaded_data(file_bytes):
    import io
    return pd.read_excel(io.BytesIO(file_bytes))

@st.cache_data
def prepare_monthly(df):
    df = df.copy()
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

# ── Header ────────────────────────────────────────
st.markdown('<p class="big-title">💰 FinAI</p>', unsafe_allow_html=True)
st.markdown('<p class="small-sub">Your personal AI financial advisor</p>', unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────
st.markdown('<p class="sec">Upload Transaction Data</p>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload your bank Excel file (.xlsx)", type=["xlsx"], label_visibility="collapsed")

if uploaded_file:
    df = load_uploaded_data(uploaded_file.read())
    st.success(f"✅ {df.shape[0]} transactions loaded")
else:
    df = load_default_data()
    st.caption("Using sample data — upload your own file for personalised results")

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

# ── Inputs ────────────────────────────────────────
st.markdown('<p class="sec">Your Finances</p>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
with c1: income   = st.number_input("Monthly Income (₹)",   min_value=0.0, value=10000.0, step=500.0)
with c2: expenses = st.number_input("Monthly Expenses (₹)", min_value=0.0, value=5000.0,  step=500.0)
with c3: savings  = st.number_input("Current Savings (₹)",  min_value=0.0, value=4000.0,  step=500.0)
with c4: debt     = st.number_input("Total Debt (₹)",       min_value=0.0, value=1000.0,  step=500.0)

# ── Goals ─────────────────────────────────────────
st.markdown('<p class="sec">Your Goals</p>', unsafe_allow_html=True)
num_goals = st.radio("Number of goals", [1, 2], horizontal=True, label_visibility="collapsed")

goals = []
cols  = st.columns(num_goals)
for i in range(num_goals):
    with cols[i]:
        st.markdown(f"**Goal {i+1}**")
        n = st.text_input(f"What is your goal?",         value="Buy a car"    if i==0 else "Trip to Europe", key=f"gn{i}")
        c = st.number_input(f"How much does it cost? (₹)", min_value=0.0, value=1500000.0 if i==0 else 200000.0, step=10000.0, key=f"gc{i}")
        m = st.number_input(f"In how many months?",        min_value=1.0, value=24.0       if i==0 else 12.0,     step=1.0,     key=f"gm{i}")
        goals.append({"name": n, "cost": c, "months": m})

st.markdown("<br>", unsafe_allow_html=True)
go = st.button("⚡ Analyse", use_container_width=True)

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

    # ── Score ──────────────────────────────────────
    st.markdown('<p class="sec">Financial Health Score</p>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(f"""
        <div class="score-wrap">
            <div class="score-num" style="color:{hex_color}">{final_score:.1f}</div>
            <div style="color:#444; font-size:14px;">out of 100</div>
            <div class="score-lbl" style="color:{hex_color}">{label}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        if   final_score >= 75: st.success("Your finances are in excellent shape!")
        elif final_score >= 50: st.info("Good shape — room to improve.")
        elif final_score >= 30: st.warning("Needs attention.")
        else:                   st.error("Needs immediate attention.")
        st.progress(int(final_score))
        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""<div class="card">
                <div class="card-val" style="color:{'#2ecc71' if er<=70 else '#e74c3c'}">{er:.1f}%</div>
                <div class="card-lbl">Expense Ratio</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="card">
                <div class="card-val" style="color:{'#2ecc71' if sr>=20 else '#e74c3c'}">{sr:.1f}%</div>
                <div class="card-lbl">Savings Rate</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""<div class="card">
                <div class="card-val" style="color:{'#2ecc71' if dti<=36 else '#e74c3c'}">{dti:.1f}%</div>
                <div class="card-lbl">Debt to Income</div>
            </div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""<div class="card">
                <div class="card-val" style="color:{'#2ecc71' if cf>=0 else '#e74c3c'}">₹{cf:,.0f}</div>
                <div class="card-lbl">Monthly Surplus</div>
            </div>""", unsafe_allow_html=True)

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
        bar_color        = "#2ecc71" if progress_pct >= 50 else "#f39c12" if progress_pct >= 25 else "#e74c3c"

        goal_results.append({
            "goal": goal, "remaining": remaining,
            "needed_per_month": needed_per_month, "achievable": achievable,
            "realistic_months": realistic_months, "completion_date": completion_date,
            "progress_pct": progress_pct
        })

        st.markdown(f"**🎯 Goal {i+1} — {goal['name']}**")
        st.markdown(f"""
        <div class="goal-wrap">
            <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
                <span style="color:#fff;font-weight:600;">{goal['name']}</span>
                <span style="color:{bar_color};font-weight:600;">{progress_pct:.1f}% complete</span>
            </div>
            <div style="background:#0f1117;border-radius:8px;height:14px;overflow:hidden;">
                <div style="background:{bar_color};width:{progress_pct}%;height:100%;border-radius:8px;"></div>
            </div>
            <div style="display:flex;justify-content:space-between;margin-top:6px;">
                <span style="color:#555;font-size:12px;">Saved: ₹{savings:,.0f}</span>
                <span style="color:#555;font-size:12px;">Target: ₹{goal['cost']:,.0f}</span>
            </div>
        </div>""", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class="card">
                <div class="card-val">₹{remaining:,.0f}</div>
                <div class="card-lbl">Amount Remaining</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="card">
                <div class="card-val" style="color:{'#2ecc71' if achievable else '#e74c3c'}">₹{needed_per_month:,.0f}</div>
                <div class="card-lbl">Needed Per Month</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="card">
                <div class="card-val">{completion_date.strftime('%b %Y')}</div>
                <div class="card-lbl">Estimated Completion</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if achievable:
            st.success(f"✅ Achievable by {target_date.strftime('%B %Y')}!")
        else:
            shortfall = needed_per_month - cf
            st.error(f"❌ Not achievable in {goal['months']:.0f} months — realistic timeline: {realistic_months:.0f} months")
            st.info(f"💡 Reduce expenses by ₹{shortfall:,.0f}/month to hit your target")

        # Simulator
        st.markdown(f"**🔧 What if you save more?**")
        extra = st.slider(f"Extra savings per month for Goal {i+1} (₹)",
                          0, int(income), 0, 500, key=f"sl{i}")
        if extra > 0:
            new_months = remaining / (cf + extra) if (cf + extra) > 0 else 9999
            new_date   = datetime.now() + relativedelta(months=int(new_months))
            saved_months = max(0, realistic_months - new_months)
            st.success(f"💡 Saving ₹{extra:,.0f} more → Goal by **{new_date.strftime('%B %Y')}** — {saved_months:.0f} months earlier!")

        # Month plan
        st.markdown(f"**📅 Month by Month Plan**")
        plan = []
        running = savings
        for m in range(1, min(int(goal['months']) + 3, 13)):
            month_save = min(needed_per_month, cf)
            running   += month_save
            pct        = min(100, running / goal['cost'] * 100)
            plan.append({
                'Month'          : f'Month {m}',
                'Save (₹)'       : f'₹{month_save:,.0f}',
                'Total Saved (₹)': f'₹{running:,.0f}',
                'Progress'       : f'{pct:.1f}%'
            })
            if running >= goal['cost']:
                break
        st.dataframe(pd.DataFrame(plan), use_container_width=True, hide_index=True)
        st.markdown("---")

    # Combined summary for 2 goals
    if len(goals) == 2:
        st.markdown('<p class="sec">Combined Goal Summary</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class="card">
                <div class="card-val">₹{sum(g['cost'] for g in goals):,.0f}</div>
                <div class="card-lbl">Total Goal Amount</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="card">
                <div class="card-val">₹{sum(r['remaining'] for r in goal_results):,.0f}</div>
                <div class="card-lbl">Total Remaining</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="card">
                <div class="card-val" style="color:{'#2ecc71' if cf>=total_needed else '#e74c3c'}">₹{total_needed:,.0f}</div>
                <div class="card-lbl">Total Needed/Month</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if cf >= total_needed:
            st.success(f"✅ Your surplus of ₹{cf:,.0f} covers both goals!")
        else:
            st.error(f"❌ You need ₹{total_needed - cf:,.0f} more per month to achieve both goals")

    # ── Budget Rule ────────────────────────────────
    st.markdown('<p class="sec">50/30/20 Budget Rule</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="goal-wrap">
            <p style="color:#aaa; font-size:13px; margin:0 0 12px;">Recommended split of ₹{income:,.0f}</p>
            <div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #1e2030;">
                <span style="color:#fff;">Needs (50%)</span>
                <span style="color:#378ADD;">₹{income*0.5:,.0f}</span>
            </div>
            <div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #1e2030;">
                <span style="color:#fff;">Wants (30%)</span>
                <span style="color:#BA7517;">₹{income*0.3:,.0f}</span>
            </div>
            <div style="display:flex;justify-content:space-between;padding:6px 0;">
                <span style="color:#fff;">Savings (20%)</span>
                <span style="color:#2ecc71;">₹{income*0.2:,.0f}</span>
            </div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if exp_gap > 0: st.error(f"❌ Reduce expenses by ₹{exp_gap:,.0f}")
        else:           st.success(f"✅ Expenses ₹{abs(exp_gap):,.0f} under limit")
        if sav_gap > 0: st.error(f"❌ Save ₹{sav_gap:,.0f} more")
        else:           st.success(f"✅ Savings ₹{abs(sav_gap):,.0f} above target")
        if ideal_score > final_score:
            st.info(f"💡 Following 50/30/20 improves your score by {ideal_score-final_score:.1f} pts")
    with c2:
        fig, ax = plt.subplots(figsize=(5, 3.5), facecolor='#16181f')
        ax.set_facecolor('#16181f')
        cats = ['Needs\n50%','Wants\n30%','Savings\n20%','Your\nExpenses','Your\nSavings']
        vals = [income*0.5, income*0.3, income*0.2, expenses, savings]
        cols = ['#378ADD','#BA7517','#2ecc71','#e74c3c','#2ecc71']
        bars = ax.bar(cats, vals, color=cols, edgecolor='#0f1117', width=0.6)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+80,
                    f'₹{val:,.0f}', ha='center', color='white', fontsize=8)
        ax.tick_params(colors='#555'); ax.set_facecolor('#16181f')
        for s in ['top','right','left']: ax.spines[s].set_visible(False)
        ax.spines['bottom'].set_color('#1e2030')
        ax.yaxis.set_visible(False)
        st.pyplot(fig); plt.close()

    # ── Monthly Trend ──────────────────────────────
    st.markdown('<p class="sec">6 Month Projection</p>', unsafe_allow_html=True)
    pe_vals = [expenses*(1-0.01*i) for i in range(1,7)]
    ps_vals = [savings *(1+0.01*i) for i in range(1,7)]
    ms      = [f'M{i}' for i in range(1,7)]
    c1, c2  = st.columns(2)
    with c1:
        rows = []
        for i in range(6):
            rows.append({'Month': ms[i], 'Expenses': f'₹{pe_vals[i]:,.0f}',
                         'Savings': f'₹{ps_vals[i]:,.0f}',
                         'Surplus': f'₹{income-pe_vals[i]:,.0f}'})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    with c2:
        fig, ax = plt.subplots(figsize=(5, 3.5), facecolor='#16181f')
        ax.set_facecolor('#16181f')
        ax.plot(ms, pe_vals, marker='o', color='#e74c3c', linewidth=2, label='Expenses')
        ax.plot(ms, ps_vals, marker='s', color='#2ecc71', linewidth=2, label='Savings')
        ax.tick_params(colors='#555')
        for s in ['top','right']: ax.spines[s].set_visible(False)
        ax.spines['bottom'].set_color('#1e2030')
        ax.spines['left'].set_color('#1e2030')
        ax.legend(facecolor='#16181f', labelcolor='white', fontsize=9)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'₹{x:,.0f}'))
        ax.tick_params(colors='#555')
        st.pyplot(fig); plt.close()

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
        fig, ax = plt.subplots(figsize=(5, 3.5), facecolor='#16181f')
        ax.set_facecolor('#16181f')
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(cat_df)))
        cat_df.plot(kind='barh', ax=ax, color=colors, edgecolor='#0f1117')
        ax.tick_params(colors='#555')
        for s in ['top','right']: ax.spines[s].set_visible(False)
        ax.spines['bottom'].set_color('#1e2030')
        ax.spines['left'].set_color('#1e2030')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'₹{x:,.0f}'))
        ax.set_xlabel(''); ax.set_ylabel('')
        st.pyplot(fig); plt.close()

    # ── AI Advisor ─────────────────────────────────
    st.markdown('<p class="sec">🤖 AI Financial Advisor</p>', unsafe_allow_html=True)
    with st.spinner("Analysing your finances..."):
        try:
            goals_text = "\n".join([
                f"- {g['goal']['name']}: ₹{g['goal']['cost']:,.0f} in {g['goal']['months']:.0f} months — needs ₹{g['needed_per_month']:,.0f}/month — {'achievable' if g['achievable'] else 'not achievable in time'}"
                for g in goal_results
            ])
            prompt = f"""
You are a friendly financial advisor. Talk directly to the user like a friend.

FINANCES:
- Income: ₹{income:,.0f} | Expenses: ₹{expenses:,.0f} | Savings: ₹{savings:,.0f} | Debt: ₹{debt:,.0f}
- Monthly Surplus: ₹{cf:,.0f} | Score: {final_score:.1f}/100 ({label})

GOALS:
{goals_text}

TOP SPENDING: {top_3_cats[0]} (₹{top_3_amounts[0]:,.0f}), {top_3_cats[1]} (₹{top_3_amounts[1]:,.0f}), {top_3_cats[2]} (₹{top_3_amounts[2]:,.0f})

Give 5 short specific tips to help achieve their goals faster.
Mention exact categories to cut and exact amounts.
Give realistic timelines. No jargon. No mention of ML or models. Use ₹.
"""
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}]
            )
            st.markdown(response.choices[0].message.content)
        except Exception as e:
            st.error(f"Could not load recommendations: {e}")

    # ── PDF ────────────────────────────────────────
    st.markdown('<p class="sec">Download Report</p>', unsafe_allow_html=True)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "FinAI — Financial Health Report", ln=True, align="C")
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Finances", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 6, f"Income: Rs {income:,.0f}  |  Expenses: Rs {expenses:,.0f}  |  Savings: Rs {savings:,.0f}  |  Debt: Rs {debt:,.0f}", ln=True)
    pdf.cell(0, 6, f"Monthly Surplus: Rs {cf:,.0f}  |  Score: {final_score:.1f} / 100 ({label})", ln=True)
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Goals", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for i, gr in enumerate(goal_results):
        pdf.cell(0, 6, f"Goal {i+1}: {gr['goal']['name']} — Rs {gr['goal']['cost']:,.0f} — {'Achievable' if gr['achievable'] else 'Needs more savings'} — Est. {gr['completion_date'].strftime('%B %Y')}", ln=True)
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Key Metrics", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 6, f"Expense Ratio: {er:.1f}%  |  Savings Rate: {sr:.1f}%  |  Debt-to-Income: {dti:.1f}%", ln=True)

    pdf_bytes = pdf.output()
    st.download_button("📥 Download Report", bytes(pdf_bytes),
                       "finai_report.pdf", "application/pdf",
                       use_container_width=True)
