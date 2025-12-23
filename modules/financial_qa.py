"""
Financial Q&A Module
====================
AI-powered question answering system for financial data analysis.
Uses Groq API to answer questions based on analyzed financial data.
"""

import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime


class FinancialQA:
    """
    AI-powered Q&A system for financial analysis.
    Processes user questions and generates contextual answers using analyzed data.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the Q&A system.
        
        Args:
            api_key: Groq API key for AI completions
        """
        self.api_key = api_key
        self.client = None
        
        if api_key:
            try:
                from groq import Groq
                self.client = Groq(api_key=api_key)
            except ImportError:
                raise ImportError("Groq package not installed. Install with: pip install groq")
    
    def prepare_financial_context(
        self,
        df: pd.DataFrame,
        kpis: pd.DataFrame,
        pnl: pd.DataFrame,
        forecasts: Dict[str, Any],
        runway: pd.DataFrame,
        recommendations: List[Dict],
        insights: List[str]
    ) -> str:
        """
        Prepare comprehensive financial context for the AI.
        
        Args:
            df: Original transaction dataframe
            kpis: Monthly KPIs dataframe
            pnl: Profit & Loss statement
            forecasts: Revenue, expenses, and profit forecasts
            runway: Cash runway projections
            recommendations: Strategic recommendations
            insights: Key insights from analysis
            
        Returns:
            Formatted context string for AI
        """
        # Basic metrics
        total_revenue = pnl['revenue'].sum()
        total_expenses = pnl['expenses'].sum()
        net_profit = pnl['net_profit'].sum()
        profit_margin = (net_profit / total_revenue * 100) if total_revenue > 0 else 0
        
        # Date range
        date_range = f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}"
        num_months = len(kpis)
        
        # Average monthly metrics
        avg_monthly_revenue = kpis['revenue'].mean()
        avg_monthly_expenses = kpis['expenses'].mean()
        avg_monthly_profit = kpis['net_profit'].mean()
        avg_burn_rate = kpis['burn_rate'].mean()
        
        # Growth trends
        revenue_trend = ((kpis['revenue'].iloc[-1] - kpis['revenue'].iloc[0]) / 
                        kpis['revenue'].iloc[0] * 100) if len(kpis) > 1 and kpis['revenue'].iloc[0] > 0 else 0
        expense_trend = ((kpis['expenses'].iloc[-1] - kpis['expenses'].iloc[0]) / 
                        kpis['expenses'].iloc[0] * 100) if len(kpis) > 1 and kpis['expenses'].iloc[0] > 0 else 0
        
        # Forecast data
        rev_forecast = forecasts['revenue']['hybrid']
        exp_forecast = forecasts['expenses']['hybrid']
        profit_forecast = forecasts['net_profit']['hybrid']
        
        forecast_rev_growth = ((rev_forecast['yhat'].iloc[-1] - rev_forecast['yhat'].iloc[0]) / 
                              rev_forecast['yhat'].iloc[0] * 100) if rev_forecast['yhat'].iloc[0] > 0 else 0
        
        # Cash runway status
        negative_months = runway[runway['cash_balance'] < 0]
        if len(negative_months) > 0:
            months_left = len(runway[runway['cash_balance'] > 0])
            runway_status = f"{months_left} months until cash depletion (CRITICAL)"
        else:
            runway_status = "Positive cash balance maintained (HEALTHY)"
        
        # Category breakdown
        category_revenue = df[df['amount'] > 0].groupby('category')['amount'].sum().sort_values(ascending=False)
        category_expenses = df[df['amount'] < 0].groupby('category')['amount'].sum().abs().sort_values(ascending=False)
        
        top_revenue_categories = category_revenue.head(5).to_dict()
        top_expense_categories = category_expenses.head(5).to_dict()
        
        # Vendor analysis
        top_vendors = df.groupby('vendor')['amount'].sum().abs().sort_values(ascending=False).head(5).to_dict()
        
        # Build context string
        context = f"""
FINANCIAL DATA ANALYSIS CONTEXT
================================

PERIOD: {date_range} ({num_months} months of data)

OVERALL PERFORMANCE
-------------------
Total Revenue: ${total_revenue:,.2f}
Total Expenses: ${total_expenses:,.2f}
Net Profit: ${net_profit:,.2f}
Profit Margin: {profit_margin:.2f}%

MONTHLY AVERAGES
----------------
Average Monthly Revenue: ${avg_monthly_revenue:,.2f}
Average Monthly Expenses: ${avg_monthly_expenses:,.2f}
Average Monthly Profit: ${avg_monthly_profit:,.2f}
Average Burn Rate: ${avg_burn_rate:,.2f}

GROWTH TRENDS (Historical)
--------------------------
Revenue Growth: {revenue_trend:+.1f}%
Expense Growth: {expense_trend:+.1f}%

6-MONTH FORECASTS
-----------------
Revenue Forecast Growth: {forecast_rev_growth:+.1f}%
Projected Monthly Revenue: ${rev_forecast['yhat'].mean():,.2f}
Projected Monthly Expenses: ${exp_forecast['yhat'].mean():,.2f}
Projected Monthly Profit: ${profit_forecast['yhat'].mean():,.2f}

CASH RUNWAY
-----------
Status: {runway_status}
Current Burn Rate: ${avg_burn_rate:,.2f}/month

TOP REVENUE CATEGORIES
----------------------
{chr(10).join([f'- {cat}: ${amt:,.2f}' for cat, amt in list(top_revenue_categories.items())[:5]])}

TOP EXPENSE CATEGORIES
----------------------
{chr(10).join([f'- {cat}: ${amt:,.2f}' for cat, amt in list(top_expense_categories.items())[:5]])}

TOP VENDORS/CUSTOMERS
---------------------
{chr(10).join([f'- {vendor}: ${amt:,.2f}' for vendor, amt in list(top_vendors.items())[:5]])}

KEY INSIGHTS
------------
{chr(10).join(['- ' + insight for insight in insights[:10]])}

STRATEGIC RECOMMENDATIONS
-------------------------
{chr(10).join([f"{rec['priority']} {rec['category']}: {rec['issue']}" for rec in recommendations[:5]])}

MONTHLY KPI DETAILS
-------------------
{kpis.to_string()}
"""
        
        return context
    
    def answer_question(
        self,
        question: str,
        context: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> str:
        """
        Answer a user's question about the financial data.
        
        Args:
            question: User's question
            context: Financial data context
            conversation_history: Previous Q&A pairs for context
            
        Returns:
            AI-generated answer
        """
        if not self.client:
            return "Error: Groq API key not configured. Please add your API key in the sidebar."
        
        try:
            # Build system prompt
            system_prompt = """You are an expert CFO and financial analyst. Your role is to answer questions about the company's financial data with precision and clarity.

Guidelines:
- Provide specific, data-driven answers using the provided financial context
- Use exact numbers and percentages from the data
- Be concise but comprehensive
- Highlight trends, patterns, and insights
- If asked about forecasts, mention they are AI-generated projections
- If the question cannot be answered with the available data, say so clearly
- Provide actionable recommendations when relevant
- Use professional financial terminology
- Format numbers with appropriate currency symbols and units
"""
            
            # Build user prompt
            user_prompt = f"""
FINANCIAL DATA CONTEXT:
{context}

USER QUESTION:
{question}

Please provide a clear, data-driven answer to the question above. Use specific numbers and insights from the financial data provided. If recommending actions, make them specific and actionable.
"""
            
            # Build messages with conversation history
            messages = [{"role": "system", "content": system_prompt}]
            
            if conversation_history:
                for entry in conversation_history[-5:]:  # Last 5 exchanges for context
                    messages.append({"role": "user", "content": entry.get("question", "")})
                    messages.append({"role": "assistant", "content": entry.get("answer", "")})
            
            messages.append({"role": "user", "content": user_prompt})
            
            # Call Groq API
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                temperature=0.2,
                max_tokens=800,
                messages=messages
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def suggest_questions(self, context: str) -> List[str]:
        """
        Generate suggested questions based on the financial data.
        
        Args:
            context: Financial data context
            
        Returns:
            List of suggested questions
        """
        default_questions = [
            "What is our current cash runway and when will we run out of cash?",
            "What are our top revenue sources and expenses?",
            "How is our profitability trending over time?",
            "What cost-cutting measures should we prioritize?",
            "What is our revenue growth rate and forecast?",
            "Which vendors or categories are driving our highest expenses?",
            "What are the biggest risks to our financial health?",
            "How can we improve our profit margins?",
            "What is our average monthly burn rate?",
            "Should we be fundraising right now?"
        ]
        
        if not self.client:
            return default_questions
        
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                temperature=0.7,
                max_tokens=400,
                messages=[
                    {"role": "system", "content": "You are a CFO. Generate 5 insightful questions that would be valuable to ask about this financial data."},
                    {"role": "user", "content": f"Based on this financial summary, what are 5 important questions to ask?\n\n{context[:1500]}\n\nReturn ONLY the questions, one per line, without numbering."}
                ]
            )
            
            questions = [q.strip() for q in response.choices[0].message.content.strip().split('\n') if q.strip()]
            return questions[:5] if questions else default_questions
            
        except Exception:
            return default_questions
