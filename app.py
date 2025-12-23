"""
AI CFO Dashboard - Complete Financial Analysis System
=====================================================
Features:
- Upload CSV/Excel files
- Automatic financial analysis
- Interactive visualizations
- PDF report generation
- AI-powered insights

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import io
import sys
import os
from pathlib import Path

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent / "modules"))

from modules.financial_loader import FinancialLoader
from modules.financial_kpis import FinancialKPIs
from modules.financial_statements import FinancialStatements
from modules.forecast_engine import HybridForecastEngine
from modules.pdf_report import generate_cfo_pdf_report
from modules.financial_qa import FinancialQA

# Page config
st.set_page_config(
    page_title="AI CFO Dashboard",
    page_icon="💰",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stAlert {
        background-color: #e3f2fd;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'groq_api_key' not in st.session_state:
    # Use Streamlit secrets first, then environment variable
    st.session_state.groq_api_key = st.secrets.get('groq_api_key', os.getenv('GROQ_API_KEY', ''))
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []
if 'financial_context' not in st.session_state:
    st.session_state.financial_context = None

# Helper Functions
def generate_ai_summary_with_groq(api_key, forecasts, runway, recommendations, kpis, pnl):
    """Generate AI-powered CFO summary using Groq API"""
    if not api_key:
        return None
    
    try:
        from groq import Groq
        
        client = Groq(api_key=api_key)
        
        # Prepare data
        rev_data = forecasts['revenue']['hybrid'][['ds', 'yhat']].head(3).to_dict('records')
        total_revenue = pnl['revenue'].sum()
        total_expenses = pnl['expenses'].sum()
        net_profit = pnl['net_profit'].sum()
        profit_margin = (net_profit / total_revenue * 100) if total_revenue > 0 else 0
        
        # Cash runway
        negative_months = runway[runway['cash_balance'] < 0]
        if len(negative_months) > 0:
            runway_status = f"CRITICAL: {len(runway[runway['cash_balance'] > 0])} months remaining"
        else:
            runway_status = "HEALTHY: Positive cash maintained"
        
        # Format recommendations
        rec_text = "\n".join([
            f"{r['priority']} {r['category']}: {r['issue']}"
            for r in recommendations[:3]  # Top 3
        ])
        
        user_prompt = f"""
You are an experienced CFO. Provide a concise executive summary for the board.

FINANCIALS:
- Revenue: ${total_revenue:,.0f}
- Expenses: ${total_expenses:,.0f}
- Net Profit: ${net_profit:,.0f}
- Margin: {profit_margin:.1f}%
- Cash Runway: {runway_status}

TOP ISSUES:
{rec_text}

Provide:
1. One-sentence financial health assessment
2. Top 3 priorities for the next 30 days
3. One key risk to monitor

Keep it under 150 words.
"""
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=300,
            messages=[
                {"role": "system", "content": "You are an experienced CFO providing concise strategic advice."},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"AI Summary Error: {str(e)}"


def load_file(uploaded_file):
    """Load CSV or Excel file"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload CSV or Excel.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None


def generate_ai_recommendations(api_key, forecasts, runway, kpis, pnl):
    """Generate AI-powered strategic recommendations"""
    if not api_key:
        return None
    
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        
        # Prepare comprehensive data for AI
        rev_forecast = forecasts['revenue']['hybrid']
        exp_forecast = forecasts['expenses']['hybrid']
        profit_forecast = forecasts['net_profit']['hybrid']
        
        revenue_growth = ((rev_forecast['yhat'].iloc[-1] - rev_forecast['yhat'].iloc[0]) / 
                         rev_forecast['yhat'].iloc[0] * 100)
        expense_growth = ((exp_forecast['yhat'].iloc[-1] - exp_forecast['yhat'].iloc[0]) / 
                         exp_forecast['yhat'].iloc[0] * 100)
        avg_profit = profit_forecast['yhat'].mean()
        
        total_revenue = pnl['revenue'].sum()
        total_expenses = pnl['expenses'].sum()
        net_profit = pnl['net_profit'].sum()
        profit_margin = (net_profit / total_revenue * 100) if total_revenue > 0 else 0
        
        # Cash runway
        negative_months = runway[runway['cash_balance'] < 0]
        if len(negative_months) > 0:
            months_left = len(runway[runway['cash_balance'] > 0])
            runway_status = f"CRITICAL: {months_left} months until cash depletion"
        else:
            runway_status = "HEALTHY: Positive cash balance maintained"
        
        # Burn rate
        avg_burn = kpis['burn_rate'].mean()
        
        user_prompt = f"""
You are an experienced CFO analyzing a company's financials. Generate 3-5 strategic recommendations with specific, actionable steps.

CURRENT FINANCIALS:
- Total Revenue: ${total_revenue:,.0f}
- Total Expenses: ${total_expenses:,.0f}
- Net Profit: ${net_profit:,.0f}
- Profit Margin: {profit_margin:.1f}%
- Average Monthly Burn: ${avg_burn:,.0f}

FORECASTS (6 months):
- Revenue Growth: {revenue_growth:+.1f}%
- Expense Growth: {expense_growth:+.1f}%
- Avg Projected Profit: ${avg_profit:,.0f}/month

CASH POSITION:
- Status: {runway_status}

For each recommendation, provide:
1. Priority level (CRITICAL/HIGH/MEDIUM/LOW)
2. Category (e.g., Revenue, Cost Control, Liquidity)
3. Specific issue description
4. 3-5 concrete action items with timelines
5. Expected impact if possible

Format as JSON array:
[
  {{
    "priority": "CRITICAL",
    "category": "Category Name",
    "issue": "Specific problem description",
    "actions": ["Action 1 with timeline", "Action 2...", ...],
    "impact": "Expected outcome"
  }}
]

Focus on the most urgent issues first. Be specific and actionable.
"""
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.4,
            max_tokens=1500,
            messages=[
                {"role": "system", "content": "You are an experienced CFO providing strategic recommendations. Always respond with valid JSON array."},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        # Parse AI response
        import json
        ai_text = response.choices[0].message.content
        
        # Extract JSON from response
        if '```json' in ai_text:
            ai_text = ai_text.split('```json')[1].split('```')[0]
        elif '```' in ai_text:
            ai_text = ai_text.split('```')[1].split('```')[0]
        
        recommendations = json.loads(ai_text.strip())
        
        # Add emoji to priority
        for rec in recommendations:
            priority = rec.get('priority', 'MEDIUM').upper()
            if 'CRITICAL' in priority:
                rec['priority'] = '🔴 CRITICAL'
            elif 'HIGH' in priority:
                rec['priority'] = '🔴 HIGH'
            elif 'MEDIUM' in priority:
                rec['priority'] = '🟡 MEDIUM'
            elif 'LOW' in priority:
                rec['priority'] = '🟢 LOW'
            else:
                rec['priority'] = '🟢 GOOD'
        
        return recommendations
        
    except Exception as e:
        print(f"AI Recommendations Error: {e}")
        return None

def generate_cfo_recommendations(forecasts, runway, kpis, pnl):
    """Generate strategic CFO recommendations for future actions"""
    recommendations = []
    
    try:
        # Revenue analysis
        rev_forecast = forecasts['revenue']['hybrid']
        revenue_growth = ((rev_forecast['yhat'].iloc[-1] - rev_forecast['yhat'].iloc[0]) / 
                         rev_forecast['yhat'].iloc[0] * 100)
        
        if revenue_growth < 0:
            recommendations.append({
                "priority": "🔴 CRITICAL",
                "category": "Revenue Recovery",
                "issue": f"Revenue declining by {abs(revenue_growth):.1f}%",
                "actions": [
                    "Conduct customer churn analysis immediately",
                    "Review and optimize pricing strategy",
                    "Implement customer retention program",
                    "Launch win-back campaign for lost customers"
                ]
            })
        elif revenue_growth < 5:
            recommendations.append({
                "priority": "🟡 MEDIUM",
                "category": "Revenue Growth",
                "issue": f"Low growth at {revenue_growth:.1f}%",
                "actions": [
                    "Increase marketing spend on high-ROI channels",
                    "Develop upsell and cross-sell programs",
                    "Explore new customer segments",
                    "Consider strategic partnerships"
                ]
            })
        else:
            recommendations.append({
                "priority": "🟢 GOOD",
                "category": "Revenue Growth",
                "issue": f"Strong growth at {revenue_growth:.1f}%",
                "actions": [
                    "Maintain momentum with current strategies",
                    "Scale successful marketing campaigns",
                    "Consider expanding team capacity",
                    "Document winning processes"
                ]
            })
        
        # Expense management
        exp_forecast = forecasts['expenses']['hybrid']
        expense_growth = ((exp_forecast['yhat'].iloc[-1] - exp_forecast['yhat'].iloc[0]) / 
                         exp_forecast['yhat'].iloc[0] * 100)
        
        if expense_growth > 10:
            recommendations.append({
                "priority": "🔴 HIGH",
                "category": "Cost Control",
                "issue": f"Expenses rising {expense_growth:.1f}%",
                "actions": [
                    "Conduct comprehensive cost audit",
                    "Renegotiate vendor contracts",
                    "Implement expense approval workflow",
                    "Identify automation opportunities"
                ]
            })
        
        # Profitability
        profit_forecast = forecasts['net_profit']['hybrid']
        avg_profit = profit_forecast['yhat'].mean()
        
        if avg_profit < 0:
            recommendations.append({
                "priority": "🔴 CRITICAL",
                "category": "Profitability",
                "issue": f"Projected loss: ${abs(avg_profit):,.0f}/month",
                "actions": [
                    "URGENT: Implement emergency cost reduction",
                    "Accelerate revenue generation initiatives",
                    "Consider strategic pivots or product changes",
                    "Prepare contingency funding plan"
                ]
            })
        elif avg_profit < 10000:
            recommendations.append({
                "priority": "🟡 MEDIUM",
                "category": "Profitability",
                "issue": f"Low margins: ${avg_profit:,.0f}/month",
                "actions": [
                    "Focus on improving gross margins",
                    "Optimize operational efficiency",
                    "Review pricing structure",
                    "Reduce customer acquisition costs"
                ]
            })
        
        # Cash runway
        negative_months = runway[runway['cash_balance'] < 0]
        if len(negative_months) > 0:
            months_left = len(runway[runway['cash_balance'] > 0])
            if months_left < 6:
                recommendations.append({
                    "priority": "🔴 CRITICAL",
                    "category": "Liquidity",
                    "issue": f"Cash depleted in {months_left} months",
                    "actions": [
                        "URGENT: Begin fundraising immediately",
                        "Cut non-essential expenses by 30%",
                        "Accelerate collections from customers",
                        "Explore bridge financing options",
                        "Prepare detailed 13-week cash flow forecast"
                    ]
                })
            elif months_left < 12:
                recommendations.append({
                    "priority": "🟡 HIGH",
                    "category": "Fundraising",
                    "issue": f"{months_left} months runway remaining",
                    "actions": [
                        "Start fundraising process (4-6 month timeline)",
                        "Update investor materials and pitch deck",
                        "Improve unit economics to attract investors",
                        "Build investor pipeline and warm leads"
                    ]
                })
        else:
            recommendations.append({
                "priority": "🟢 GOOD",
                "category": "Cash Management",
                "issue": "Healthy cash position",
                "actions": [
                    "Maintain 12+ months runway target",
                    "Invest excess cash in growth initiatives",
                    "Build strategic reserves for opportunities",
                    "Monitor burn rate monthly"
                ]
            })
        
        # Strategic priorities
        total_revenue = pnl['revenue'].sum()
        total_expenses = pnl['expenses'].sum()
        
        if total_revenue > 0 and total_expenses > 0:
            efficiency_ratio = total_expenses / total_revenue
            if efficiency_ratio > 0.8:
                recommendations.append({
                    "priority": "🟡 MEDIUM",
                    "category": "Operational Efficiency",
                    "issue": f"Operating ratio at {efficiency_ratio:.1%}",
                    "actions": [
                        "Target 70% or better operating efficiency",
                        "Implement process automation",
                        "Review team productivity metrics",
                        "Eliminate redundant tools and subscriptions"
                    ]
                })
        
    except Exception as e:
        recommendations.append({
            "priority": "ℹ️ INFO",
            "category": "System",
            "issue": "Error generating some recommendations",
            "actions": ["Review data quality and completeness"]
        })
    
    return recommendations

def generate_ai_insights(kpis, pnl, variance, runway):
    """Generate AI-powered insights"""
    insights = []
    
    # Revenue insights
    latest_revenue = pnl['revenue'].iloc[-1] if len(pnl) > 0 else 0
    avg_revenue = pnl['revenue'].mean() if len(pnl) > 0 else 0
    
    if latest_revenue > avg_revenue * 1.1:
        insights.append("🟢 Revenue is above average - strong performance trend")
    elif latest_revenue < avg_revenue * 0.9:
        insights.append("🔴 Revenue below average - investigate decline factors")
    
    # Burn rate insights
    avg_burn = kpis['burn_rate'].mean() if len(kpis) > 0 else 0
    if avg_burn > 0:
        insights.append(f"⚠️ Average burn rate: ${avg_burn:,.2f}/month - monitor closely")
    
    # Profitability insights
    profitable_months = len(pnl[pnl['net_profit'] > 0]) if len(pnl) > 0 else 0
    total_months = len(pnl) if len(pnl) > 0 else 1
    
    if profitable_months / total_months > 0.7:
        insights.append("🟢 Strong profitability - 70%+ months profitable")
    elif profitable_months / total_months < 0.3:
        insights.append("🔴 Profitability concern - less than 30% months profitable")
    
    # Runway insights
    if len(runway) > 0:
        negative_months = runway[runway['cash_balance'] < 0]
        if len(negative_months) > 0:
            first_negative = negative_months.iloc[0]['ds']
            insights.append(f"⚠️ Cash runway critical - funds depleted by {first_negative.strftime('%Y-%m')}")
        else:
            insights.append("🟢 Cash runway healthy - no immediate liquidity risk")
    
    return insights

def plot_revenue_expenses(pnl):
    """Plot revenue vs expenses"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    months = [str(m) for m in pnl.index]
    x = range(len(months))
    
    ax.bar([i - 0.2 for i in x], pnl['revenue'], width=0.4, label='Revenue', color='#2ecc71')
    ax.bar([i + 0.2 for i in x], pnl['expenses'], width=0.4, label='Expenses', color='#e74c3c')
    
    ax.set_xlabel('Month')
    ax.set_ylabel('Amount ($)')
    ax.set_title('Revenue vs Expenses Over Time')
    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_profit_trend(pnl):
    """Plot net profit trend"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    months = [str(m) for m in pnl.index]
    profits = pnl['net_profit']
    colors = ['#2ecc71' if p > 0 else '#e74c3c' for p in profits]
    
    ax.bar(months, profits, color=colors)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Month')
    ax.set_ylabel('Net Profit ($)')
    ax.set_title('Net Profit Trend')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_forecast(forecast_results):
    """Plot revenue/expense forecast"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    titles = ['Revenue Forecast', 'Expense Forecast', 'Net Profit Forecast']
    keys = ['revenue', 'expenses', 'net_profit']
    
    for ax, title, key in zip(axes, titles, keys):
        data = forecast_results[key]
        history = data['history']
        forecast = data['hybrid']
        
        ax.plot(history['ds'], history['y'], label='Historical', marker='o')
        ax.plot(forecast['ds'], forecast['yhat'], label='Forecast', marker='s', linestyle='--')
        ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                        alpha=0.3, label='Confidence Interval')
        
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel('Amount ($)')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig

def plot_cash_runway(runway):
    """Plot cash runway"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#2ecc71' if cash > 0 else '#e74c3c' for cash in runway['cash_balance']]
    ax.bar(runway['ds'], runway['cash_balance'], color=colors)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    ax.set_xlabel('Month')
    ax.set_ylabel('Cash Balance ($)')
    ax.set_title('Cash Runway Projection')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig

# Main App
st.markdown('<p class="main-header">💰 AI CFO Dashboard</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📊 Controls")
    
    # AI Status
    if st.session_state.groq_api_key:
        st.success("✅ AI Features Enabled")
    else:
        st.warning("⚠️ AI features unavailable - API key not configured")
    
    st.markdown("---")
    
    uploaded_file = st.file_uploader(
        "Upload Financial Data",
        type=['csv', 'xlsx', 'xls'],
        help="Upload CSV or Excel file with columns: date, description, amount, category, vendor"
    )
    
    st.markdown("---")
    
    if uploaded_file:
        if st.button("🔍 Analyze Data", type="primary"):
            with st.spinner("Analyzing financial data..."):
                # Load file
                raw_df = load_file(uploaded_file)
                
                if raw_df is not None:
                    try:
                        # Process with FinancialLoader
                        loader = FinancialLoader()
                        
                        # Save temporarily to load with loader
                        temp_path = "temp_upload.csv"
                        if uploaded_file.name.endswith('.csv'):
                            raw_df.to_csv(temp_path, index=False)
                        else:
                            raw_df.to_csv(temp_path, index=False)
                        
                        st.session_state.df = loader.load_csv(temp_path)
                        st.session_state.analysis_complete = True
                        st.success("✅ Analysis complete!")
                        
                        # Clean up temp file
                        import os
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        
                    except Exception as e:
                        st.error(f"Error processing data: {str(e)}")
    
    st.markdown("---")
    st.markdown("### 📋 Sample Data Format")
    st.code("""
date,description,amount,category,vendor
2024-01-15,Product Sale,5000,Revenue,ClientA
2024-01-20,Server Costs,-200,IT,AWS
2024-02-10,Consulting,8000,Revenue,ClientB
    """)

# Main Content
if st.session_state.analysis_complete and st.session_state.df is not None:
    df = st.session_state.df
    
    # Run analyses
    kpi_engine = FinancialKPIs()
    kpis = kpi_engine.monthly_kpis(df)
    
    statements = FinancialStatements()
    pnl = statements.generate_pnl(df)
    variance = statements.variance_analysis(pnl)
    
    forecast_engine = HybridForecastEngine(periods=6)
    forecasts = forecast_engine.forecast_all(df)
    
    profit_forecast = forecasts['net_profit']['hybrid']
    runway = forecast_engine.compute_cash_runway(profit_forecast, starting_cash=300000)
    
    # AI Insights
    st.header("🤖 AI-Powered Insights")
    insights = generate_ai_insights(kpis, pnl, variance, runway)
    
    cols = st.columns(2)
    for idx, insight in enumerate(insights):
        with cols[idx % 2]:
            st.info(insight)
    
    # AI Executive Summary
    if st.session_state.groq_api_key:
        st.header("🧠 AI Executive Summary")
        with st.spinner("Generating AI summary..."):
            # Try AI recommendations first
            ai_recommendations = generate_ai_recommendations(
                st.session_state.groq_api_key,
                forecasts,
                runway,
                kpis,
                pnl
            )
            
            # If AI recommendations fail, use rule-based
            if not ai_recommendations:
                recommendations = generate_cfo_recommendations(forecasts, runway, kpis, pnl)
            else:
                recommendations = ai_recommendations
            
            ai_summary = generate_ai_summary_with_groq(
                st.session_state.groq_api_key,
                forecasts,
                runway,
                recommendations,
                kpis,
                pnl
            )
            if ai_summary and not ai_summary.startswith("AI Summary Error"):
                st.success(ai_summary)
            elif ai_summary:
                st.warning(ai_summary)
                st.info("💡 Tip: Check your API key or get one at https://console.groq.com")
    else:
        st.info("💡 Add your Groq API key in the sidebar to enable AI-powered executive summaries")
        recommendations = None
    
    # CFO Recommendations
    st.header("💼 CFO Strategic Recommendations")
    
    # Show AI badge if using AI
    if st.session_state.groq_api_key and 'ai_recommendations' in locals() and ai_recommendations:
        st.markdown("**🤖 AI-Generated strategic recommendations tailored to your business**")
    else:
        st.markdown("**📊 Rule-based recommendations (Add API key for AI-powered insights)**")
    
    if 'recommendations' not in locals() or recommendations is None:
        recommendations = generate_cfo_recommendations(forecasts, runway, kpis, pnl)
    
    # Sort by priority
    priority_order = {'🔴 CRITICAL': 0, '🔴 HIGH': 1, '🟡 MEDIUM': 2, '🟡 HIGH': 1, '🟢 GOOD': 3, 'ℹ️ INFO': 4}
    recommendations.sort(key=lambda x: priority_order.get(x['priority'], 5))
    
    for rec in recommendations:
        with st.expander(f"{rec['priority']} - {rec['category']}: {rec['issue']}", expanded=(rec['priority'].startswith('🔴'))):
            st.markdown("**Recommended Actions:**")
            for action in rec['actions']:
                st.markdown(f"• {action}")
            
            # Show expected impact if AI-generated
            if 'impact' in rec and rec['impact']:
                st.markdown(f"**Expected Impact:** {rec['impact']}")
    
    # Key Metrics
    st.header("📈 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_revenue = df[df['type'] == 'Revenue']['amount'].sum()
        st.metric("Total Revenue", f"${total_revenue:,.2f}")
    
    with col2:
        total_expenses = abs(df[df['type'] == 'Expense']['amount'].sum())
        st.metric("Total Expenses", f"${total_expenses:,.2f}")
    
    with col3:
        net_profit = total_revenue - total_expenses
        st.metric("Net Profit", f"${net_profit:,.2f}", 
                 delta=f"{(net_profit/total_revenue*100):.1f}%" if total_revenue > 0 else "0%")
    
    with col4:
        avg_burn = kpis['burn_rate'].mean()
        runway_months = kpis['estimated_runway_months'].iloc[0] if len(kpis) > 0 else 0
        if pd.notna(runway_months):
            st.metric("Cash Runway", f"{runway_months:.1f} months")
        else:
            st.metric("Cash Runway", "Healthy")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "📈 Forecasts", "💰 P&L Statement", "📉 Variance", "🔍 Raw Data"
    ])
    
    with tab1:
        st.subheader("Revenue vs Expenses")
        fig1 = plot_revenue_expenses(pnl)
        st.pyplot(fig1)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Net Profit Trend")
            fig2 = plot_profit_trend(pnl)
            st.pyplot(fig2)
        
        with col2:
            st.subheader("Monthly KPIs")
            st.dataframe(kpis.reset_index(), width='stretch')
    
    with tab2:
        st.subheader("6-Month Forecast")
        fig3 = plot_forecast(forecasts)
        st.pyplot(fig3)
        
        st.subheader("Cash Runway Projection")
        fig4 = plot_cash_runway(runway)
        st.pyplot(fig4)
    
    with tab3:
        st.subheader("Profit & Loss Statement")
        st.dataframe(pnl.reset_index(), width='stretch')
    
    with tab4:
        st.subheader("Month-over-Month Variance Analysis")
        st.dataframe(variance.reset_index(), width='stretch')
    
    with tab5:
        st.subheader("Transaction Data")
        st.dataframe(df, width='stretch')
        
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Processed Data",
            data=csv,
            file_name=f"financial_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    # Q&A Section
    st.header("💬 Ask Questions About Your Data")
    
    if st.session_state.groq_api_key:
        # Initialize Q&A system
        qa_system = FinancialQA(st.session_state.groq_api_key)
        
        # Prepare financial context if not already done
        if st.session_state.financial_context is None:
            with st.spinner("Preparing financial context..."):
                st.session_state.financial_context = qa_system.prepare_financial_context(
                    df=df,
                    kpis=kpis,
                    pnl=pnl,
                    forecasts=forecasts,
                    runway=runway,
                    recommendations=recommendations,
                    insights=insights
                )
        
        # Suggested questions
        with st.expander("💡 Suggested Questions", expanded=False):
            suggested = qa_system.suggest_questions(st.session_state.financial_context)
            for i, suggestion in enumerate(suggested):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"{i+1}. {suggestion}")
                with col2:
                    if st.button("Ask", key=f"suggest_{i}"):
                        st.session_state.current_question = suggestion
                        st.rerun()
        
        # Question input
        question = st.text_input(
            "Ask a question about your financial data:",
            value=st.session_state.get('current_question', ''),
            placeholder="e.g., What is our cash runway? How can we improve profitability?",
            key="qa_input"
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            ask_button = st.button("🔍 Get Answer", type="primary", use_container_width=True)
        with col2:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.qa_history = []
                st.session_state.current_question = ""
                st.rerun()
        
        # Process question
        if ask_button and question:
            with st.spinner("Analyzing your question..."):
                answer = qa_system.answer_question(
                    question=question,
                    context=st.session_state.financial_context,
                    conversation_history=st.session_state.qa_history
                )
                
                # Add to history
                st.session_state.qa_history.append({
                    "question": question,
                    "answer": answer,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                # Clear current question
                if 'current_question' in st.session_state:
                    del st.session_state.current_question
        
        # Display conversation history
        if st.session_state.qa_history:
            st.subheader("📝 Conversation History")
            
            # Reverse to show latest first
            for idx, entry in enumerate(reversed(st.session_state.qa_history)):
                with st.container():
                    st.markdown(f"**Q{len(st.session_state.qa_history) - idx}:** {entry['question']}")
                    st.markdown(f"**A:** {entry['answer']}")
                    st.caption(f"🕒 {entry['timestamp']}")
                    st.markdown("---")
        
        # Download Q&A history
        if st.session_state.qa_history:
            qa_text = "\n\n".join([
                f"Q: {entry['question']}\nA: {entry['answer']}\nTime: {entry['timestamp']}"
                for entry in st.session_state.qa_history
            ])
            st.download_button(
                label="📥 Download Q&A History",
                data=qa_text,
                file_name=f"qa_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    else:
        st.info("💡 Add your Groq API key in the sidebar to enable AI-powered Q&A about your financial data")
        st.markdown("""
        **What you can ask:**
        - Cash runway and liquidity questions
        - Revenue and expense analysis
        - Profitability insights
        - Cost-cutting recommendations
        - Growth forecasts
        - Strategic financial advice
        
        Get your free API key at [Groq Console](https://console.groq.com)
        """)
    
    # PDF Report Generation
    st.header("📄 Generate Report")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        report_title = st.text_input("Report Title", "AI CFO Financial Report")
    
    with col2:
        if st.button("📄 Generate PDF Report", type="primary"):
            with st.spinner("Generating PDF report..."):
                try:
                    # Generate AI summary if API key available
                    if st.session_state.groq_api_key:
                        ai_summary = generate_ai_summary_with_groq(
                            st.session_state.groq_api_key,
                            forecasts,
                            runway,
                            recommendations,
                            kpis,
                            pnl
                        )
                        if ai_summary and not ai_summary.startswith("AI Summary Error"):
                            llm_summary = f"""
AI-Powered Executive Summary
============================

{ai_summary}

Financial Metrics
-----------------
Period: {df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}

Total Revenue: ${total_revenue:,.2f}
Total Expenses: ${total_expenses:,.2f}
Net Profit: ${net_profit:,.2f}
Profit Margin: {(net_profit/total_revenue*100):.1f}%

Key Insights:
{chr(10).join(['- ' + insight for insight in insights])}
"""
                        else:
                            llm_summary = f"""
Executive Summary
-----------------
Period: {df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}

Total Revenue: ${total_revenue:,.2f}
Total Expenses: ${total_expenses:,.2f}
Net Profit: ${net_profit:,.2f}
Profit Margin: {(net_profit/total_revenue*100):.1f}%

Key Insights:
{chr(10).join(['- ' + insight for insight in insights])}

Financial Health: {'Strong' if net_profit > 0 else 'Requires Attention'}
"""
                    else:
                        llm_summary = f"""
Executive Summary
-----------------
Period: {df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}

Total Revenue: ${total_revenue:,.2f}
Total Expenses: ${total_expenses:,.2f}
Net Profit: ${net_profit:,.2f}
Profit Margin: {(net_profit/total_revenue*100):.1f}%

Key Insights:
{chr(10).join(['- ' + insight for insight in insights])}

Financial Health: {'Strong' if net_profit > 0 else 'Requires Attention'}

Note: Add Groq API key for AI-powered insights
                    """
                    
                    # Generate PDF
                    pdf_path = f"reports/cfo_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    
                    # Create reports directory if it doesn't exist
                    Path("reports").mkdir(exist_ok=True)
                    
                    generate_cfo_pdf_report(
                        file_path=pdf_path,
                        llm_summary=llm_summary,
                        kpis=kpis,
                        pnl=pnl,
                        variance=variance,
                        runway=runway,
                        recommendations=recommendations
                    )
                    
                    with open(pdf_path, "rb") as pdf_file:
                        st.download_button(
                            label="📥 Download PDF Report",
                            data=pdf_file,
                            file_name=pdf_path.split('/')[-1],
                            mime="application/pdf"
                        )
                    
                    st.success("✅ PDF report generated successfully!")
                    
                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")

else:
    # Welcome screen
    st.markdown("""
    ## Welcome to AI CFO Dashboard! 👋
    
    This comprehensive financial analysis system helps you:
    
    - 📊 **Analyze** your financial data automatically
    - 📈 **Forecast** revenue, expenses, and cash runway
    - 💰 **Track** KPIs and profitability trends
    - 📄 **Generate** professional PDF reports
    - 🤖 **Get** AI-powered insights and recommendations
    
    ### Getting Started
    
    1. Upload your financial data (CSV or Excel) using the sidebar
    2. Click "Analyze Data" to process your information
    3. Explore interactive visualizations and insights
    4. Generate and download PDF reports
    
    ### Required Data Format
    
    Your file should contain these columns:
    - `date`: Transaction date
    - `description`: Transaction description
    - `amount`: Transaction amount (positive for revenue, negative for expenses)
    - `category`: Transaction category
    - `vendor`: Vendor/customer name
    
    ### Try it now! 
    Upload your data in the sidebar to get started. 👈
    """)
    
    # Sample data preview
    st.subheader("📋 Sample Data Preview")
    sample_data = pd.DataFrame({
        'date': ['2024-01-15', '2024-01-20', '2024-02-10', '2024-02-15'],
        'description': ['Product Sale', 'Server Costs', 'Consulting', 'Marketing'],
        'amount': [5000, -200, 8000, -500],
        'category': ['Revenue', 'IT', 'Revenue', 'Marketing'],
        'vendor': ['ClientA', 'AWS', 'ClientB', 'GoogleAds']
    })
    st.dataframe(sample_data, width='stretch')