"""
AI CFO – Fixed Complete Financial Pipeline
==========================================
Loads financial data, computes KPIs, performs forecasting,
evaluates runway, generates recommendations, and creates reports.

FIXES APPLIED:
- Fixed Prophet dependency issues (using hybrid ML approach)
- Fixed month handling consistency
- Fixed ARIMA forecast indexing
- Added proper error handling
- Compatible with Windows + Python 3.12

Dependencies:
    pip install pandas matplotlib statsmodels scikit-learn fpdf2 groq openpyxl
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent / "modules"))

from modules.financial_loader import FinancialLoader
from modules.financial_kpis import FinancialKPIs
from modules.financial_statements import FinancialStatements
from modules.forecast_engine import HybridForecastEngine
from modules.pdf_report import generate_cfo_pdf_report


# ---------------------------
# Rule-Based CFO Recommendations
# ---------------------------

def generate_cfo_recommendations(forecast_results, runway, kpis):
    """
    Generate strategic CFO recommendations based on financial analysis
    """
    recommendations = []
    
    try:
        # Revenue analysis
        rev_forecast = forecast_results["revenue"]["hybrid"]
        revenue_growth = ((rev_forecast["yhat"].iloc[-1] - rev_forecast["yhat"].iloc[0]) / 
                         rev_forecast["yhat"].iloc[0] * 100)
        
        if revenue_growth < 0:
            recommendations.append({
                "priority": "HIGH",
                "category": "Revenue",
                "issue": f"Revenue declining by {abs(revenue_growth):.1f}%",
                "action": "Implement immediate revenue recovery plan: analyze churn, review pricing strategy, optimize sales funnel"
            })
        elif revenue_growth < 5:
            recommendations.append({
                "priority": "MEDIUM",
                "category": "Revenue",
                "issue": f"Low revenue growth at {revenue_growth:.1f}%",
                "action": "Focus on customer acquisition and upsell opportunities"
            })
        else:
            recommendations.append({
                "priority": "LOW",
                "category": "Revenue",
                "issue": f"Strong revenue growth at {revenue_growth:.1f}%",
                "action": "Maintain momentum and consider scaling operations"
            })
        
        # Expense analysis
        exp_forecast = forecast_results["expenses"]["hybrid"]
        expense_growth = ((exp_forecast["yhat"].iloc[-1] - exp_forecast["yhat"].iloc[0]) / 
                         exp_forecast["yhat"].iloc[0] * 100)
        
        if expense_growth > 10:
            recommendations.append({
                "priority": "HIGH",
                "category": "Expenses",
                "issue": f"Expenses rising rapidly at {expense_growth:.1f}%",
                "action": "Conduct cost audit, renegotiate vendor contracts, implement cost controls"
            })
        elif expense_growth > 5:
            recommendations.append({
                "priority": "MEDIUM",
                "category": "Expenses",
                "issue": f"Moderate expense growth at {expense_growth:.1f}%",
                "action": "Monitor spending trends and identify automation opportunities"
            })
        
        # Profitability analysis
        profit_forecast = forecast_results["net_profit"]["hybrid"]
        avg_profit = profit_forecast["yhat"].mean()
        
        if avg_profit < 0:
            recommendations.append({
                "priority": "CRITICAL",
                "category": "Profitability",
                "issue": f"Average projected loss: ${abs(avg_profit):,.2f}/month",
                "action": "URGENT: Implement burn rate reduction plan immediately"
            })
        elif avg_profit < 10000:
            recommendations.append({
                "priority": "MEDIUM",
                "category": "Profitability",
                "issue": f"Low profitability: ${avg_profit:,.2f}/month",
                "action": "Improve margins through operational efficiency"
            })
        else:
            recommendations.append({
                "priority": "LOW",
                "category": "Profitability",
                "issue": f"Healthy profit margins: ${avg_profit:,.2f}/month",
                "action": "Consider reinvestment in growth initiatives"
            })
        
        # Cash runway analysis
        negative_months = runway[runway["cash_balance"] < 0]
        
        if len(negative_months) > 0:
            first_negative = negative_months.iloc[0]["ds"]
            months_to_zero = len(runway[runway["cash_balance"] > 0])
            
            if months_to_zero < 6:
                recommendations.append({
                    "priority": "CRITICAL",
                    "category": "Liquidity",
                    "issue": f"Cash depleted by {first_negative.strftime('%Y-%m')} ({months_to_zero} months)",
                    "action": "URGENT: Raise capital or implement aggressive cost reduction"
                })
            elif months_to_zero < 12:
                recommendations.append({
                    "priority": "HIGH",
                    "category": "Liquidity",
                    "issue": f"Limited runway: {months_to_zero} months until cash depletion",
                    "action": "Begin fundraising process and improve unit economics"
                })
        else:
            recommendations.append({
                "priority": "LOW",
                "category": "Liquidity",
                "issue": "Healthy cash runway",
                "action": "Maintain cash reserves and monitor burn rate"
            })
        
        # Burn rate analysis
        avg_burn = kpis["burn_rate"].mean()
        if avg_burn > 0:
            recommendations.append({
                "priority": "MEDIUM",
                "category": "Burn Rate",
                "issue": f"Average burn: ${avg_burn:,.2f}/month",
                "action": "Track burn efficiency and optimize customer acquisition cost"
            })
        
    except Exception as e:
        recommendations.append({
            "priority": "INFO",
            "category": "System",
            "issue": f"Error generating some recommendations: {str(e)}",
            "action": "Review data quality and completeness"
        })
    
    return recommendations


# ---------------------------
# LLM Summary (Groq API)
# ---------------------------

def generate_cfo_llm_summary(api_key, forecast_results, runway, recommendations, kpis, pnl):
    """
    Generate board-ready CFO summary using Groq API
    """
    try:
        from groq import Groq
        
        client = Groq(api_key=api_key)
        
        # Prepare data summaries
        rev_data = forecast_results["revenue"]["hybrid"][["ds", "yhat"]].to_dict("records")
        exp_data = forecast_results["expenses"]["hybrid"][["ds", "yhat"]].to_dict("records")
        prof_data = forecast_results["net_profit"]["hybrid"][["ds", "yhat"]].to_dict("records")
        
        # Cash runway status
        negative_months = runway[runway["cash_balance"] < 0]
        if len(negative_months) > 0:
            cash_out_date = negative_months.iloc[0]["ds"].strftime("%Y-%m")
            runway_status = f"CRITICAL: Cash depleted by {cash_out_date}"
        else:
            runway_status = "HEALTHY: Positive cash balance maintained"
        
        # Format recommendations
        rec_text = "\n".join([
            f"[{r['priority']}] {r['category']}: {r['issue']} → {r['action']}"
            for r in recommendations
        ])
        
        # Historical performance
        total_revenue = pnl["revenue"].sum()
        total_expenses = pnl["expenses"].sum()
        net_profit = pnl["net_profit"].sum()
        profit_margin = (net_profit / total_revenue * 100) if total_revenue > 0 else 0
        
        system_prompt = """
You are an experienced CFO preparing a board presentation. Create a concise, 
strategic financial summary with clear insights and actionable recommendations.
Use professional language and focus on business impact.
"""
        
        user_prompt = f"""
FINANCIAL ANALYSIS SUMMARY
==========================

HISTORICAL PERFORMANCE:
- Total Revenue: ${total_revenue:,.2f}
- Total Expenses: ${total_expenses:,.2f}
- Net Profit: ${net_profit:,.2f}
- Profit Margin: {profit_margin:.1f}%

REVENUE FORECAST (6 months):
{rev_data[:3]}

EXPENSE FORECAST (6 months):
{exp_data[:3]}

PROFIT FORECAST (6 months):
{prof_data[:3]}

CASH RUNWAY STATUS:
{runway_status}

KEY RECOMMENDATIONS:
{rec_text}

Generate a structured board-ready summary with:
1. Executive Summary (2-3 sentences)
2. Revenue & Growth Analysis
3. Expense Management Insights
4. Profitability Outlook
5. Cash Flow & Runway Assessment
6. Top 3 Strategic Priorities (30/60/90 day plan)
7. Risk Factors
8. CFO Recommendation

Keep it concise and strategic. Focus on actionable insights.
"""
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=2000,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"""
FINANCIAL ANALYSIS SUMMARY
==========================

Executive Summary:
Analysis complete. Review detailed metrics below for financial performance insights.

Note: LLM summary unavailable - {str(e)}

Please review the generated recommendations and financial statements for detailed analysis.
"""


# ---------------------------
# Visualization Functions
# ---------------------------

def create_financial_charts(pnl, forecasts, runway, output_dir="reports/charts"):
    """
    Create comprehensive financial charts
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    charts = {}
    
    # 1. Revenue vs Expenses
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    months = [str(m) for m in pnl.index]
    x = np.arange(len(months))
    
    ax1.bar(x - 0.2, pnl['revenue'], width=0.4, label='Revenue', color='#2ecc71')
    ax1.bar(x + 0.2, pnl['expenses'], width=0.4, label='Expenses', color='#e74c3c')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Amount ($)')
    ax1.set_title('Revenue vs Expenses')
    ax1.set_xticks(x)
    ax1.set_xticklabels(months, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    chart1_path = f"{output_dir}/revenue_expenses.png"
    plt.savefig(chart1_path, dpi=300, bbox_inches='tight')
    charts['revenue_expenses'] = chart1_path
    plt.close()
    
    # 2. Profit Trend
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    colors = ['#2ecc71' if p > 0 else '#e74c3c' for p in pnl['net_profit']]
    ax2.bar(months, pnl['net_profit'], color=colors)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Net Profit ($)')
    ax2.set_title('Net Profit Trend')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    chart2_path = f"{output_dir}/profit_trend.png"
    plt.savefig(chart2_path, dpi=300, bbox_inches='tight')
    charts['profit_trend'] = chart2_path
    plt.close()
    
    # 3. Forecast Chart
    fig3, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, (key, title) in zip(axes, [
        ('revenue', 'Revenue Forecast'),
        ('expenses', 'Expense Forecast'),
        ('net_profit', 'Net Profit Forecast')
    ]):
        hist = forecasts[key]['history']
        fc = forecasts[key]['hybrid']
        
        ax.plot(hist['ds'], hist['y'], 'o-', label='Historical', linewidth=2)
        ax.plot(fc['ds'], fc['yhat'], 's--', label='Forecast', linewidth=2)
        ax.fill_between(fc['ds'], fc['yhat_lower'], fc['yhat_upper'], 
                        alpha=0.3, label='Confidence')
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel('Amount ($)')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    chart3_path = f"{output_dir}/forecast.png"
    plt.savefig(chart3_path, dpi=300, bbox_inches='tight')
    charts['forecast'] = chart3_path
    plt.close()
    
    # 4. Cash Runway
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    colors = ['#2ecc71' if cash > 0 else '#e74c3c' for cash in runway['cash_balance']]
    ax4.bar(runway['ds'], runway['cash_balance'], color=colors)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax4.set_xlabel('Month')
    ax4.set_ylabel('Cash Balance ($)')
    ax4.set_title('Cash Runway Projection')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    chart4_path = f"{output_dir}/cash_runway.png"
    plt.savefig(chart4_path, dpi=300, bbox_inches='tight')
    charts['cash_runway'] = chart4_path
    plt.close()
    
    return charts


# ---------------------------
# Main Pipeline
# ---------------------------

def run_cfo_pipeline(data_path, api_key=None, starting_cash=300000):
    """
    Execute complete CFO analysis pipeline
    """
    print("\n" + "="*60)
    print("AI CFO FINANCIAL ANALYSIS PIPELINE")
    print("="*60 + "\n")
    
    try:
        # 1. Load Data
        print("📊 Loading financial data...")
        loader = FinancialLoader()
        df = loader.load_csv(data_path)
        print(f"   ✓ Loaded {len(df)} transactions")
        print(f"   ✓ Period: {df['date'].min()} to {df['date'].max()}")
        
        # 2. Compute KPIs
        print("\n📈 Computing KPIs...")
        kpi_engine = FinancialKPIs()
        kpis = kpi_engine.monthly_kpis(df)
        print(f"   ✓ Analyzed {len(kpis)} months")
        
        # 3. Generate Statements
        print("\n💰 Generating financial statements...")
        statements = FinancialStatements()
        pnl = statements.generate_pnl(df)
        variance = statements.variance_analysis(pnl)
        print("   ✓ P&L and variance analysis complete")
        
        # 4. Forecast
        print("\n🔮 Running forecast models...")
        forecast_engine = HybridForecastEngine(periods=6)
        forecasts = forecast_engine.forecast_all(df)
        print("   ✓ 6-month forecast complete (Hybrid ML + ARIMA)")
        
        # 5. Cash Runway
        print("\n💵 Computing cash runway...")
        profit_forecast = forecasts["net_profit"]["hybrid"]
        runway = forecast_engine.compute_cash_runway(profit_forecast, starting_cash)
        
        negative_months = runway[runway["cash_balance"] < 0]
        if len(negative_months) > 0:
            months_left = len(runway[runway["cash_balance"] > 0])
            print(f"   ⚠️  WARNING: Cash depleted in {months_left} months")
        else:
            print("   ✓ Cash runway healthy")
        
        # 6. Generate Recommendations
        print("\n🎯 Generating strategic recommendations...")
        recommendations = generate_cfo_recommendations(forecasts, runway, kpis)
        print(f"   ✓ {len(recommendations)} recommendations generated")
        
        # Print recommendations
        print("\n" + "-"*60)
        print("STRATEGIC RECOMMENDATIONS")
        print("-"*60)
        for rec in sorted(recommendations, key=lambda x: 
                         ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO'].index(x['priority'])):
            print(f"\n[{rec['priority']}] {rec['category']}")
            print(f"  Issue: {rec['issue']}")
            print(f"  Action: {rec['action']}")
        
        # 7. LLM Summary
        if api_key:
            print("\n\n🤖 Generating AI CFO summary...")
            llm_summary = generate_cfo_llm_summary(api_key, forecasts, runway, 
                                                   recommendations, kpis, pnl)
        else:
            print("\n\n⚠️  Skipping AI summary (no API key provided)")
            llm_summary = "AI summary unavailable - no API key provided"
        
        print("\n" + "="*60)
        print("CFO EXECUTIVE SUMMARY")
        print("="*60)
        print(llm_summary)
        
        # 8. Generate Charts
        print("\n\n📊 Creating visualizations...")
        charts = create_financial_charts(pnl, forecasts, runway)
        print(f"   ✓ {len(charts)} charts generated")
        
        # 9. Generate PDF Report
        print("\n📄 Generating PDF report...")
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = f"reports/cfo_report_{timestamp}.pdf"
        
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
        
        print(f"   ✓ PDF report saved: {pdf_path}")
        
        print("\n" + "="*60)
        print("✅ ANALYSIS COMPLETE")
        print("="*60)
        
        return {
            'df': df,
            'kpis': kpis,
            'pnl': pnl,
            'variance': variance,
            'forecasts': forecasts,
            'runway': runway,
            'recommendations': recommendations,
            'llm_summary': llm_summary,
            'charts': charts,
            'pdf_path': pdf_path
        }
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# ---------------------------
# Run Pipeline
# ---------------------------

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='AI CFO Financial Analysis')
    parser.add_argument('--data', type=str, default='data/mixed_business_financials.csv',
                       help='Path to financial data CSV')
    parser.add_argument('--api-key', type=str, default=None,
                       help='Groq API key for LLM summary')
    parser.add_argument('--cash', type=float, default=300000,
                       help='Starting cash balance')
    
    args = parser.parse_args()
    
    # Run pipeline
    results = run_cfo_pipeline(
        data_path=args.data,
        api_key=args.api_key or os.getenv('GROQ_API_KEY'),
        starting_cash=args.cash
    )
    
    if results:
        print(f"\n📊 Results saved in 'reports/' directory")
        print(f"📄 PDF Report: {results['pdf_path']}")