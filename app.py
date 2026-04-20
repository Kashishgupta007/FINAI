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

st.set_page_config(page_title="My Financial Health", page_icon="💰", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #0f1117; }
.main-title { font-size: 42px; font-weight: 800; color: #ffffff; text-align: center; padding: 20px 0 5px 0; }
.sub-title { font-size: 16px; color: #888; text-align: center; margin-bottom: 30px; }
.score-box { background: #1e3a5f; border-radius: 20px; padding: 30px; text-align: center; border: 1px solid #1e4d8c; }
.score-number { font-size: 72px; font-weight: 900; line-height: 1; }
.score-label { font-size: 22px; font-weight: 600; margin-top: 8px; }
.metric-card { background: #1a1d27; border-radius: 15px; padding: 20px; border: 1px solid #2a2d3a; text-align: center; }
.metric-value { font-size: 28px; font-weight: 700; color: #ffffff; }
.metric-label { font-size: 13px; color: #888; margin-top: 4px; }
.section-header { font-size: 22px; font-weight: 700; color: #ffffff; margin: 30px 0 15px 0; padding-left: 10px; border-left: 4px solid #378ADD; }
.goal-card { background: #1a1d27; border-radius: 15px; padding: 20px; border: 1px solid #2a2d3a; margin-bottom: 15px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">💰 My Financial Health</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Tell us your goals and we will help you achieve them</p>', unsafe_allow_html=True)

# ── File Upload ───────────────────────────────────
st.markdown('<p class="section-header">Upload Your Transaction Data</p>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload your bank transaction file (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.success(f"✅ File uploaded — {df.shape[0]} transactions found")
else:
    df = pd.read_excel("personal_transactions_dashboard_ready (2).xlsx")
    st.info("ℹ️ Using sample data — upload your own bank file above")

# ── Train Models ──────────────────────────────────
df['Month'] = pd.to_datetime(df['Month'])

monthly = df.groupby('Month').apply(lambda g: pd.Series({
    'income'  : g[g['Transaction Type'] == 'credit']['Amount'].sum(),
    'expenses': g[g['Transaction Type'] == 'debit' ]['Amount'].sum(),
})).reset_index()

monthly['savings'] = monthly['income'] - monthly['expenses']
monthly['debt']    = monthly['expenses'] * 0.10

def compute_health_score(income, expenses, savings, debt):
    score = 0
    exp_ratio = expenses / income if income > 0 else 1
    sav_rate  = savings  / income if income > 0 else 0
    dti       = debt     / income if income > 0 else 1
    if   exp_ratio <= 0.50: score += 40
    elif exp_ratio <= 0.70: score += 30
    elif exp_ratio <= 0.90: score += 15
    if   sav_rate >= 0.30: score += 35
    elif sav_rate >= 0.20: score += 28
    elif sav_rate >= 0.10: score += 15
    elif sav_rate >= 0.00: score +=  5
    if   dti <= 0.15: score += 25
    elif dti <= 0.36: score += 18
    elif dti <= 0.50: score +=  8
    return min(100, score)

monthly['health_score'] = monthly.apply(
    lambda r: compute_health_score(r['income'], r['expenses'], r['savings'], r['debt']), axis=1)

X = monthly[['income','expenses','savings','debt']].values
y = monthly['health_score'].values

lr_model  = LinearRegression()
lr_model.fit(X, y)

xgb_model = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
xgb_model.fit(X, y)

avg_score    = monthly['health_score'].mean()
best_months  = monthly[monthly['health_score'] >= 75]
avg_exp_best = best_months['expenses'].mean() if len(best_months) > 0 else 0
avg_sav_best = best_months['savings'].mean()  if len(best_months) > 0 else 0
cat_spending = df[df['Transaction Type'] == 'debit'].groupby('Category')['Amount'].sum().sort_values(ascending=False)
top_3_cats   = list(cat_spending.head(3).index)
top_3_amounts = list(cat_spending.head(3).values)

# ── Basic Questions ───────────────────────────────
st.markdown('<p class="section-header">Tell Us About Your Finances</p>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    income   = st.number_input("1️⃣ What is your monthly income? (₹)",   min_value=0.0, value=10000.0, step=500.0)
    expenses = st.number_input("2️⃣ What are your monthly expenses? (₹)", min_value=0.0, value=5000.0,  step=500.0)
with col2:
    savings  = st.number_input("3️⃣ How much have you saved so far? (₹)", min_value=0.0, value=4000.0,  step=500.0)
    debt     = st.number_input("4️⃣ How much debt do you have? (₹)",      min_value=0.0, value=1000.0,  step=500.0)

# ── Goal Section ──────────────────────────────────
st.markdown('<p class="section-header">Your Financial Goals</p>', unsafe_allow_html=True)
st.caption("Add up to 2 goals — we will build a personalised plan to help you achieve them")

num_goals = st.radio("How many goals do you have?", [1, 2], horizontal=True)

goals = []
col1, col2 = st.columns(2)

with col1:
    st.markdown("**🎯 Goal 1**")
    g1_name     = st.text_input("What is your goal?", value="Buy a car", key="g1_name")
    g1_cost     = st.number_input("How much does it cost? (₹)", min_value=0.0, value=1500000.0, step=10000.0, key="g1_cost")
    g1_months   = st.number_input("In how many months do you want it?", min_value=1.0, value=24.0, step=1.0, key="g1_months")
    goals.append({"name": g1_name, "cost": g1_cost, "months": g1_months})

if num_goals == 2:
    with col2:
        st.markdown("**🎯 Goal 2**")
        g2_name   = st.text_input("What is your goal?", value="Trip to Europe", key="g2_name")
        g2_cost   = st.number_input("How much does it cost? (₹)", min_value=0.0, value=200000.0, step=10000.0, key="g2_cost")
        g2_months = st.number_input("In how many months do you want it?", min_value=1.0, value=12.0, step=1.0, key="g2_months")
        goals.append({"name": g2_name, "cost": g2_cost, "months": g2_months})

st.markdown("<br>", unsafe_allow_html=True)
calculate = st.button("⚡ Analyse My Finances & Goals", use_container_width=True)

if calculate:
    cf = income - expenses

    lr_score  = float(np.clip(lr_model.predict([[income, expenses, savings, debt]])[0],  0, 100))
    xgb_score = float(np.clip(xgb_model.predict([[income, expenses, savings, debt]])[0], 0, 100))
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

    exp_gap = expenses - (income * 0.50)
    sav_gap = (income * 0.20) - savings

    ideal_expenses = income * 0.50
    ideal_savings  = income * 0.20
    ideal_lr  = float(np.clip(lr_model.predict([[income, ideal_expenses, ideal_savings, debt]])[0],  0, 100))
    ideal_xgb = float(np.clip(xgb_model.predict([[income, ideal_expenses, ideal_savings, debt]])[0], 0, 100))
    ideal_score = (ideal_lr + ideal_xgb) / 2

    # ── Score ──────────────────────────────────────
    st.markdown('<p class="section-header">Your Financial Health Score</p>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"""
        <div class="score-box">
            <div class="score-number" style="color:{hex_color};">{final_score:.1f}</div>
            <div style="color:#888; font-size:18px;">out of 100</div>
            <div class="score-label" style="color:{hex_color};">{label}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if final_score >= 75:
            st.success("Your finances are in excellent shape!")
        elif final_score >= 50:
            st.info("Your finances are decent but there is room to improve.")
        elif final_score >= 30:
            st.warning("Your finances need some attention.")
        else:
            st.error("Your finances need immediate attention.")
        st.progress(int(final_score))
        st.caption("Score calculated using Linear Regression + XGBoost trained on your transaction data")

    # ── Goal Analysis ──────────────────────────────
    st.markdown('<p class="section-header">🎯 Goal Analysis</p>', unsafe_allow_html=True)

    total_monthly_needed = 0
    goal_results = []

    for i, goal in enumerate(goals):
        remaining       = max(0, goal['cost'] - savings)
        needed_per_month = remaining / goal['months'] if goal['months'] > 0 else remaining
        total_monthly_needed += needed_per_month
        achievable      = cf >= needed_per_month
        realistic_months = remaining / cf if cf > 0 else 9999
        completion_date = datetime.now() + relativedelta(months=int(realistic_months))
        target_date     = datetime.now() + relativedelta(months=int(goal['months']))
        progress_pct    = min(100, (savings / goal['cost']) * 100) if goal['cost'] > 0 else 0

        # LR predicted score if they save enough
        new_expenses_lr = expenses - max(0, needed_per_month - cf)
        lr_goal_score   = float(np.clip(lr_model.predict([[income, new_expenses_lr, needed_per_month, debt]])[0], 0, 100))
        xgb_goal_score  = float(np.clip(xgb_model.predict([[income, new_expenses_lr, needed_per_month, debt]])[0], 0, 100))
        goal_score      = (lr_goal_score + xgb_goal_score) / 2

        goal_results.append({
            "goal": goal,
            "remaining": remaining,
            "needed_per_month": needed_per_month,
            "achievable": achievable,
            "realistic_months": realistic_months,
            "completion_date": completion_date,
            "target_date": target_date,
            "progress_pct": progress_pct,
            "goal_score": goal_score
        })

        st.markdown(f"### {'🎯' if i==0 else '🎯'} Goal {i+1} — {goal['name']}")

        # Progress bar
        bar_color = "#2ecc71" if progress_pct >= 50 else "#f39c12" if progress_pct >= 25 else "#e74c3c"
        st.markdown(f"""
        <div style="background:#1a1d27; border-radius:15px; padding:20px; border:1px solid #2a2d3a; margin-bottom:15px;">
            <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                <span style="color:#ffffff; font-weight:600;">{goal['name']}</span>
                <span style="color:{bar_color}; font-weight:600;">{progress_pct:.1f}% complete</span>
            </div>
            <div style="background:#2a2d3a; border-radius:10px; height:20px; overflow:hidden;">
                <div style="background:{bar_color}; width:{progress_pct}%; height:100%; border-radius:10px;"></div>
            </div>
            <div style="display:flex; justify-content:space-between; margin-top:8px;">
                <span style="color:#888; font-size:13px;">Current: ₹{savings:,.0f}</span>
                <span style="color:#888; font-size:13px;">Target: ₹{goal['cost']:,.0f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">₹{goal['cost']:,.0f}</div>
                <div class="metric-label">Goal Amount</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">₹{remaining:,.0f}</div>
                <div class="metric-label">Amount Remaining</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="color:{'#2ecc71' if achievable else '#e74c3c'};">₹{needed_per_month:,.0f}</div>
                <div class="metric-label">Needed Per Month</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if achievable:
            st.success(f"✅ Great news! You can achieve '{goal['name']}' in {goal['months']:.0f} months!")
            st.info(f"📅 Estimated completion: {target_date.strftime('%B %Y')}")
        else:
            st.error(f"❌ '{goal['name']}' is not achievable in {goal['months']:.0f} months with current savings")
            st.warning(f"⏰ Realistic timeline: {realistic_months:.0f} months — by {completion_date.strftime('%B %Y')}")
            shortfall = needed_per_month - cf
            st.info(f"💡 You need ₹{shortfall:,.0f} more per month — try reducing expenses by ₹{shortfall:,.0f}")

        # Live simulator
        st.markdown(f"**🔧 Simulator — What if you save more for '{goal['name']}'?**")
        extra_saving = st.slider(
            f"Increase monthly savings by (₹) for Goal {i+1}",
            min_value=0, max_value=int(income),
            value=0, step=500, key=f"slider_{i}"
        )
        if extra_saving > 0:
            new_cf           = cf + extra_saving
            new_months       = remaining / new_cf if new_cf > 0 else 9999
            new_date         = datetime.now() + relativedelta(months=int(new_months))
            months_saved     = realistic_months - new_months
            st.success(f"💡 Saving ₹{extra_saving:,.0f} more per month → Goal achieved by **{new_date.strftime('%B %Y')}** — {months_saved:.0f} months earlier!")

        # Month by month plan
        st.markdown(f"**📅 Month by Month Savings Plan for '{goal['name']}'**")
        plan = []
        running_savings = savings
        for m in range(1, min(int(goal['months']) + 3, 13)):
            month_save = min(needed_per_month, cf)
            running_savings += month_save
            progress = min(100, (running_savings / goal['cost']) * 100)
            plan.append({
                'Month'              : f'Month {m}',
                'Save This Month (₹)': f'₹{month_save:,.0f}',
                'Total Saved (₹)'    : f'₹{running_savings:,.0f}',
                'Progress'           : f'{progress:.1f}%'
            })
            if running_savings >= goal['cost']:
                break
        plan_df = pd.DataFrame(plan)
        st.dataframe(plan_df, use_container_width=True, hide_index=True)

        st.markdown("---")

    # Combined goal summary
    if len(goals) == 2:
        st.markdown('<p class="section-header">📊 Combined Goal Summary</p>', unsafe_allow_html=True)
        total_cost = sum(g['cost'] for g in goals)
        total_remaining = sum(r['remaining'] for r in goal_results)
        if cf >= total_monthly_needed:
            st.success(f"✅ Your monthly surplus of ₹{cf:,.0f} is enough to achieve both goals!")
        else:
            shortfall = total_monthly_needed - cf
            st.error(f"❌ You need ₹{shortfall:,.0f} more per month to achieve both goals simultaneously")
            st.info(f"💡 Consider prioritising one goal first or extending the timeline")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">₹{total_cost:,.0f}</div>
                <div class="metric-label">Total Goal Amount</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">₹{total_remaining:,.0f}</div>
                <div class="metric-label">Total Remaining</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="color:{'#2ecc71' if cf >= total_monthly_needed else '#e74c3c'};">₹{total_monthly_needed:,.0f}</div>
                <div class="metric-label">Total Needed Per Month</div>
            </div>""", unsafe_allow_html=True)

    # ── Key Metrics ────────────────────────────────
    st.markdown('<p class="section-header">Key Metrics</p>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:{'#2ecc71' if er<=70 else '#e74c3c'}">{er:.1f}%</div>
            <div class="metric-label">Expense Ratio</div>
            <div class="metric-label">{'✅ Good' if er<=70 else '❌ High'}</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:{'#2ecc71' if sr>=20 else '#e74c3c'}">{sr:.1f}%</div>
            <div class="metric-label">Savings Rate</div>
            <div class="metric-label">{'✅ Good' if sr>=20 else '❌ Low'}</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:{'#2ecc71' if dti<=36 else '#e74c3c'}">{dti:.1f}%</div>
            <div class="metric-label">Debt to Income</div>
            <div class="metric-label">{'✅ Safe' if dti<=36 else '❌ High'}</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:{'#2ecc71' if cf>=0 else '#e74c3c'}">₹{cf:,.0f}</div>
            <div class="metric-label">Monthly Surplus</div>
            <div class="metric-label">{'✅ Positive' if cf>=0 else '❌ Negative'}</div>
        </div>""", unsafe_allow_html=True)

    # ── Budget Rule ────────────────────────────────
    st.markdown('<p class="section-header">50/30/20 Budget Rule</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Recommended split of your ₹{income:,.0f} income:**")
        st.write(f"- Needs   (50%) → ₹{income*0.50:,.0f}")
        st.write(f"- Wants   (30%) → ₹{income*0.30:,.0f}")
        st.write(f"- Savings (20%) → ₹{income*0.20:,.0f}")
        st.markdown("<br>", unsafe_allow_html=True)
        if exp_gap > 0:
            st.error(f"❌ Reduce expenses by ₹{exp_gap:,.0f}")
        else:
            st.success(f"✅ Expenses are ₹{abs(exp_gap):,.0f} under the limit")
        if sav_gap > 0:
            st.error(f"❌ Save ₹{sav_gap:,.0f} more to reach 20% target")
        else:
            st.success(f"✅ Savings are ₹{abs(sav_gap):,.0f} above target")
        if ideal_score > final_score:
            st.info(f"💡 Following 50/30/20 improves your score by {ideal_score - final_score:.1f} points")
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='#1a1d27')
        ax.set_facecolor('#1a1d27')
        cats = ['Needs\n50%', 'Wants\n30%', 'Savings\n20%', 'Your\nExpenses', 'Your\nSavings']
        vals = [income*0.5, income*0.3, income*0.2, expenses, savings]
        cols = ['#378ADD', '#BA7517', '#2ecc71', '#e74c3c', '#2ecc71']
        bars = ax.bar(cats, vals, color=cols, edgecolor='#0f1117', linewidth=1.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                    f'₹{val:,.0f}', ha='center', color='white', fontsize=8)
        ax.set_title('Budget Rule vs Your Spending', color='white', fontsize=12)
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('#2a2d3a')
        ax.spines['left'].set_color('#2a2d3a')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        plt.close()

    # ── Monthly Prediction ─────────────────────────
    st.markdown('<p class="section-header">What Happens If You Improve Each Month</p>', unsafe_allow_html=True)
    future = []
    for i in range(1, 7):
        pe   = expenses * (1 - 0.01 * i)
        ps   = savings  * (1 + 0.01 * i)
        pd_v = debt     * (1 - 0.01 * i)
        pcf  = income   - pe
        future.append({
            'Month'               : f'Month {i}',
            'Expenses (₹)'        : f'₹{pe:,.0f}',
            'Savings (₹)'         : f'₹{ps:,.0f}',
            'Monthly Surplus (₹)' : f'₹{pcf:,.0f}',
        })
    future_df = pd.DataFrame(future)
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(future_df, use_container_width=True, hide_index=True)
    with col2:
        pe_vals = [expenses * (1 - 0.01 * i) for i in range(1, 7)]
        ps_vals = [savings  * (1 + 0.01 * i) for i in range(1, 7)]
        months  = [f'Month {i}' for i in range(1, 7)]
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='#1a1d27')
        ax.set_facecolor('#1a1d27')
        ax.plot(months, pe_vals, marker='o', color='#e74c3c', linewidth=2, label='Expenses')
        ax.plot(months, ps_vals, marker='s', color='#2ecc71', linewidth=2, label='Savings')
        ax.set_title('Expenses vs Savings Over 6 Months', color='white', fontsize=12)
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('#2a2d3a')
        ax.spines['left'].set_color('#2a2d3a')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(facecolor='#1a1d27', labelcolor='white')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'₹{x:,.0f}'))
        st.pyplot(fig)
        plt.close()

    # ── Category Spending ──────────────────────────
    st.markdown('<p class="section-header">Where Is Your Money Going?</p>', unsafe_allow_html=True)
    cat_df = df[df['Transaction Type'] == 'debit'].groupby('Category')['Amount'].sum().sort_values(ascending=False).head(10)
    col1, col2 = st.columns(2)
    with col1:
        cat_display = cat_df.reset_index()
        cat_display.columns = ['Category', 'Total Spent (₹)']
        cat_display['Total Spent (₹)'] = cat_display['Total Spent (₹)'].apply(lambda x: f'₹{x:,.0f}')
        st.dataframe(cat_display, use_container_width=True, hide_index=True)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='#1a1d27')
        ax.set_facecolor('#1a1d27')
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(cat_df)))
        cat_df.plot(kind='barh', ax=ax, color=colors, edgecolor='#0f1117')
        ax.set_title('Top 10 Spending Categories', color='white', fontsize=12)
        ax.set_xlabel('Total Amount (₹)', color='white')
        ax.tick_params(colors='white')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'₹{x:,.0f}'))
        ax.spines['bottom'].set_color('#2a2d3a')
        ax.spines['left'].set_color('#2a2d3a')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        plt.close()

    # ── AI Advisor ─────────────────────────────────
    st.markdown('<p class="section-header">🤖 Your Personal Financial Advisor</p>', unsafe_allow_html=True)
    with st.spinner("Analysing your finances and goals..."):
        try:
            goals_text = "\n".join([
                f"- Goal {i+1}: {g['goal']['name']} costing ₹{g['goal']['cost']:,.0f} in {g['goal']['months']:.0f} months — needs ₹{g['needed_per_month']:,.0f}/month — {'achievable' if g['achievable'] else 'not achievable in time'}"
                for i, g in enumerate(goal_results)
            ])
            prompt = f"""
You are a friendly personal financial advisor talking directly to a user.

THEIR FINANCES:
- Monthly Income   : ₹{income:,.0f}
- Monthly Expenses : ₹{expenses:,.0f}
- Current Savings  : ₹{savings:,.0f}
- Total Debt       : ₹{debt:,.0f}
- Monthly Surplus  : ₹{cf:,.0f}
- Financial Score  : {final_score:.1f} / 100 ({label})

THEIR GOALS:
{goals_text}

SPENDING PATTERNS FROM THEIR DATA:
- Top 3 categories: {top_3_cats}
- Top 3 amounts: ₹{top_3_amounts[0]:,.0f}, ₹{top_3_amounts[1]:,.0f}, ₹{top_3_amounts[2]:,.0f}
- Historical best months avg savings: ₹{avg_sav_best:,.0f}
- Historical best months avg expenses: ₹{avg_exp_best:,.0f}

Give 5 specific friendly tips to help them achieve their goals faster.
Mention exactly which categories to cut and by how much.
Tell them the realistic timeline for each goal.
Talk like a friend. Use ₹ for amounts. No jargon. No mention of ML or models.
"""
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}]
            )
            st.markdown(response.choices[0].message.content)
        except Exception as e:
            st.error(f"Could not load recommendations: {e}")

    # ── PDF ────────────────────────────────────────
    st.markdown('<p class="section-header">Download Your Report</p>', unsafe_allow_html=True)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "My Financial Health Report", ln=True, align="C")
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 9, "My Finances", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, f"Monthly Income   : Rs {income:,.0f}", ln=True)
    pdf.cell(0, 7, f"Monthly Expenses : Rs {expenses:,.0f}", ln=True)
    pdf.cell(0, 7, f"Current Savings  : Rs {savings:,.0f}", ln=True)
    pdf.cell(0, 7, f"Total Debt       : Rs {debt:,.0f}", ln=True)
    pdf.cell(0, 7, f"Monthly Surplus  : Rs {cf:,.0f}", ln=True)
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 9, f"Financial Health Score : {final_score:.1f} / 100 ({label})", ln=True)
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 9, "My Goals", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for i, gr in enumerate(goal_results):
        pdf.cell(0, 7, f"Goal {i+1} : {gr['goal']['name']}", ln=True)
        pdf.cell(0, 7, f"  Cost            : Rs {gr['goal']['cost']:,.0f}", ln=True)
        pdf.cell(0, 7, f"  Remaining       : Rs {gr['remaining']:,.0f}", ln=True)
        pdf.cell(0, 7, f"  Needed/Month    : Rs {gr['needed_per_month']:,.0f}", ln=True)
        pdf.cell(0, 7, f"  Status          : {'Achievable' if gr['achievable'] else 'Needs more savings'}", ln=True)
        pdf.cell(0, 7, f"  Completion Date : {gr['completion_date'].strftime('%B %Y')}", ln=True)
        pdf.ln(2)
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 9, "Key Metrics", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, f"Expense Ratio  : {er:.1f}%", ln=True)
    pdf.cell(0, 7, f"Savings Rate   : {sr:.1f}%", ln=True)
    pdf.cell(0, 7, f"Debt-to-Income : {dti:.1f}%", ln=True)

    pdf_bytes = pdf.output()
    st.download_button(
        label="📥 Download My Report",
        data=bytes(pdf_bytes),
        file_name="my_financial_health_report.pdf",
        mime="application/pdf",
        use_container_width=True
    )
