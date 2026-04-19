import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from fpdf import FPDF
from groq import Groq
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
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">💰 My Financial Health</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Answer 6 simple questions and we will tell you exactly where you stand</p>', unsafe_allow_html=True)

# ── File Upload ───────────────────────────────────
st.markdown('<p class="section-header">Upload Your Transaction Data</p>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload your bank transaction file (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.success(f"✅ File uploaded — {df.shape[0]} transactions found")
else:
    df = pd.read_excel("personal_transactions_dashboard_ready (2).xlsx")
    st.info("ℹ️ Using sample data — upload your own bank file above for personalised results")

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

# ── Questions ─────────────────────────────────────
st.markdown('<p class="section-header">Tell Us About Your Finances</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    income   = st.number_input("1️⃣ What is your monthly income? (₹)",   min_value=0.0, value=10000.0, step=500.0)
    expenses = st.number_input("2️⃣ What are your monthly expenses? (₹)", min_value=0.0, value=5000.0,  step=500.0)
    savings  = st.number_input("3️⃣ How much have you saved so far? (₹)", min_value=0.0, value=4000.0,  step=500.0)
with col2:
    debt     = st.number_input("4️⃣ How much debt do you have? (₹)",      min_value=0.0, value=1000.0,  step=500.0)
    goal     = st.selectbox("5️⃣ What is your financial goal?", [
        "Save more money",
        "Pay off debt",
        "Start investing",
        "Build emergency fund",
        "Improve monthly budget"
    ])

    # Dynamic goal question
    if goal == "Save more money":
        goal_target = st.number_input("6️⃣ How much do you want to save per month? (₹)", min_value=0.0, value=2000.0, step=500.0)
    elif goal == "Pay off debt":
        goal_target = st.number_input("6️⃣ In how many months do you want to clear your debt?", min_value=1.0, value=12.0, step=1.0)
    elif goal == "Start investing":
        goal_target = st.number_input("6️⃣ How much do you want to invest per month? (₹)", min_value=0.0, value=1000.0, step=500.0)
    elif goal == "Build emergency fund":
        goal_target = st.number_input("6️⃣ What is your emergency fund target? (₹)", min_value=0.0, value=30000.0, step=1000.0)
    else:
        goal_target = st.number_input("6️⃣ What should your monthly expenses be? (₹)", min_value=0.0, value=income*0.5, step=500.0)

st.markdown("<br>", unsafe_allow_html=True)
calculate = st.button("⚡ Check My Financial Health", use_container_width=True)

if calculate:

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
    cf  = income   - expenses

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
            st.success("Your finances are in excellent shape! Keep it up.")
        elif final_score >= 50:
            st.info("Your finances are decent but there is room to improve.")
        elif final_score >= 30:
            st.warning("Your finances need some attention. Follow the tips below.")
        else:
            st.error("Your finances need immediate attention. Start with reducing expenses.")
        st.progress(int(final_score))
        st.caption(f"Score is calculated based on your income, expenses, savings and debt patterns")

    # ── Key Metrics ────────────────────────────────
    st.markdown('<p class="section-header">Key Metrics</p>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:{'#2ecc71' if er<=70 else '#e74c3c'}">{er:.1f}%</div>
            <div class="metric-label">Expense Ratio</div>
            <div class="metric-label">{'✅ Good (under 70%)' if er<=70 else '❌ High (should be under 70%)'}</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:{'#2ecc71' if sr>=20 else '#e74c3c'}">{sr:.1f}%</div>
            <div class="metric-label">Savings Rate</div>
            <div class="metric-label">{'✅ Good (above 20%)' if sr>=20 else '❌ Low (should be above 20%)'}</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:{'#2ecc71' if dti<=36 else '#e74c3c'}">{dti:.1f}%</div>
            <div class="metric-label">Debt to Income</div>
            <div class="metric-label">{'✅ Safe (under 36%)' if dti<=36 else '❌ High (should be under 36%)'}</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:{'#2ecc71' if cf>=0 else '#e74c3c'}">₹{cf:,.0f}</div>
            <div class="metric-label">Monthly Surplus</div>
            <div class="metric-label">{'✅ Positive' if cf>=0 else '❌ Negative'}</div>
        </div>""", unsafe_allow_html=True)

    # ── Goal Analysis ──────────────────────────────
    st.markdown('<p class="section-header">Goal Analysis</p>', unsafe_allow_html=True)

    if goal == "Save more money":
        achievable = cf >= goal_target
        months_to_goal = goal_target * 12 / cf if cf > 0 else 999
        if achievable:
            st.success(f"✅ Yes! You can save ₹{goal_target:,.0f} per month — you currently have a surplus of ₹{cf:,.0f}")
        else:
            shortfall = goal_target - cf
            st.error(f"❌ You need ₹{shortfall:,.0f} more surplus to save ₹{goal_target:,.0f} per month")
            st.info(f"💡 Try reducing your expenses by ₹{shortfall:,.0f} to reach your goal")

    elif goal == "Pay off debt":
        months_needed = debt / cf if cf > 0 else 999
        if months_needed <= goal_target:
            st.success(f"✅ Yes! At your current surplus of ₹{cf:,.0f}/month, you can pay off ₹{debt:,.0f} in {months_needed:.0f} months — within your {goal_target:.0f} month target")
        else:
            st.error(f"❌ At your current rate it will take {months_needed:.0f} months — you wanted {goal_target:.0f} months")
            needed_cf = debt / goal_target
            st.info(f"💡 You need a monthly surplus of ₹{needed_cf:,.0f} to clear debt in {goal_target:.0f} months. Try reducing expenses by ₹{needed_cf - cf:,.0f}")

    elif goal == "Start investing":
        achievable = cf >= goal_target
        if achievable:
            st.success(f"✅ Yes! You can invest ₹{goal_target:,.0f} per month — your surplus is ₹{cf:,.0f}")
            st.info(f"💡 ₹{goal_target:,.0f}/month in a mutual fund SIP can grow significantly over 10 years")
        else:
            shortfall = goal_target - cf
            st.error(f"❌ You need ₹{shortfall:,.0f} more surplus to invest ₹{goal_target:,.0f} per month")
            st.info(f"💡 Start small — even ₹500/month in a SIP is a great start")

    elif goal == "Build emergency fund":
        months_to_goal = (goal_target - savings) / cf if cf > 0 and savings < goal_target else 0
        if savings >= goal_target:
            st.success(f"✅ You already have ₹{savings:,.0f} saved — your emergency fund target of ₹{goal_target:,.0f} is met!")
        else:
            remaining = goal_target - savings
            st.warning(f"You need ₹{remaining:,.0f} more to reach your emergency fund target")
            st.info(f"💡 At your current surplus of ₹{cf:,.0f}/month, you will reach your target in {months_to_goal:.0f} months")

    else:
        if expenses <= goal_target:
            st.success(f"✅ Your current expenses of ₹{expenses:,.0f} are already under your target of ₹{goal_target:,.0f}")
        else:
            excess = expenses - goal_target
            st.error(f"❌ You are spending ₹{excess:,.0f} more than your target of ₹{goal_target:,.0f}")
            st.info(f"💡 Review your top spending categories to cut ₹{excess:,.0f}")

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
            st.error(f"❌ Reduce expenses by ₹{exp_gap:,.0f} to stay within limit")
        else:
            st.success(f"✅ Expenses are ₹{abs(exp_gap):,.0f} under the limit")
        if sav_gap > 0:
            st.error(f"❌ Save ₹{sav_gap:,.0f} more to reach 20% target")
        else:
            st.success(f"✅ Savings are ₹{abs(sav_gap):,.0f} above target")
        st.markdown("<br>", unsafe_allow_html=True)
        if ideal_score > final_score:
            st.info(f"💡 If you follow the 50/30/20 rule your health score improves by {ideal_score - final_score:.1f} points")
        else:
            st.success("✅ You are already doing better than the 50/30/20 rule!")
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
    st.caption("This shows what your savings and expenses could look like if you improve by 1% each month")
    future = []
    for i in range(1, 7):
        pe    = expenses * (1 - 0.01 * i)
        ps    = savings  * (1 + 0.01 * i)
        pd_v  = debt     * (1 - 0.01 * i)
        pcf   = income   - pe
        future.append({
            'Month'           : f'Month {i}',
            'Expenses (₹)'    : f'₹{pe:,.0f}',
            'Savings (₹)'     : f'₹{ps:,.0f}',
            'Monthly Surplus (₹)' : f'₹{pcf:,.0f}',
        })
    future_df = pd.DataFrame(future)

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(future_df, use_container_width=True, hide_index=True)
    with col2:
        pe_vals  = [expenses * (1 - 0.01 * i) for i in range(1, 7)]
        ps_vals  = [savings  * (1 + 0.01 * i) for i in range(1, 7)]
        months   = [f'Month {i}' for i in range(1, 7)]
        fig, ax  = plt.subplots(figsize=(6, 4), facecolor='#1a1d27')
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
    with st.spinner("Analysing your finances..."):
        try:
            prompt = f"""
You are a friendly personal financial advisor talking directly to a user.
Their goal is: "{goal}" and their target is {goal_target}.

THEIR FINANCES:
- Monthly Income   : ₹{income:,.0f}
- Monthly Expenses : ₹{expenses:,.0f}
- Total Savings    : ₹{savings:,.0f}
- Total Debt       : ₹{debt:,.0f}
- Monthly Surplus  : ₹{cf:,.0f}
- Financial Score  : {final_score:.1f} / 100 ({label})
- Expense Ratio    : {er:.1f}%
- Savings Rate     : {sr:.1f}%

THEIR SPENDING PATTERNS:
- Top 3 categories they spend on: {top_3_cats}
- Historical best months avg savings: ₹{avg_sav_best:,.0f}
- Historical best months avg expenses: ₹{avg_exp_best:,.0f}

Give 5 short, friendly, specific tips to help them achieve their goal.
Talk directly to them like a friend. Use ₹ for amounts.
No financial jargon. No mention of machine learning or models.
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
    pdf.cell(0, 9, "Your Details", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, f"Monthly Income   : Rs {income:,.0f}", ln=True)
    pdf.cell(0, 7, f"Monthly Expenses : Rs {expenses:,.0f}", ln=True)
    pdf.cell(0, 7, f"Total Savings    : Rs {savings:,.0f}", ln=True)
    pdf.cell(0, 7, f"Total Debt       : Rs {debt:,.0f}", ln=True)
    pdf.cell(0, 7, f"Goal             : {goal}", ln=True)
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 9, f"Financial Health Score : {final_score:.1f} / 100 ({label})", ln=True)
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 9, "Key Metrics", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, f"Expense Ratio  : {er:.1f}%", ln=True)
    pdf.cell(0, 7, f"Savings Rate   : {sr:.1f}%", ln=True)
    pdf.cell(0, 7, f"Debt-to-Income : {dti:.1f}%", ln=True)
    pdf.cell(0, 7, f"Monthly Surplus: Rs {cf:,.0f}", ln=True)
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 9, "50/30/20 Budget Rule", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, f"Recommended Needs   (50%) : Rs {income*0.50:,.0f}", ln=True)
    pdf.cell(0, 7, f"Recommended Wants   (30%) : Rs {income*0.30:,.0f}", ln=True)
    pdf.cell(0, 7, f"Recommended Savings (20%) : Rs {income*0.20:,.0f}", ln=True)

    pdf_bytes = pdf.output()
    st.download_button(
        label="📥 Download My Report",
        data=bytes(pdf_bytes),
        file_name="my_financial_health_report.pdf",
        mime="application/pdf",
        use_container_width=True
    )
