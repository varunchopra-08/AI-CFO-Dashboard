# 💰 AI CFO - Intelligent Financial Analysis Dashboard

> **Transform your financial data into actionable insights with AI-powered CFO recommendations**

An advanced financial analysis platform that combines machine learning forecasting, real-time analytics, and AI-powered strategic recommendations to help businesses make data-driven financial decisions.

![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## 🌟 Key Features

### 📊 **Comprehensive Financial Analysis**
- **Automated Data Processing**: Upload CSV or Excel files with automatic categorization
- **Monthly KPIs**: Revenue, expenses, net profit, burn rate, and cash runway
- **P&L Statements**: Automated profit & loss generation with variance analysis
- **Interactive Visualizations**: Beautiful charts and graphs powered by Matplotlib & Seaborn

### 🤖 **AI-Powered Intelligence**
- **AI Executive Summary**: Board-ready financial analysis using Llama 3.3 70B
- **Smart Recommendations**: Context-aware strategic advice with timelines and expected impact
- **Predictive Insights**: Automatic pattern detection and risk identification
- **Natural Language Analysis**: AI understands your business context and provides personalized guidance
- **💬 AI Q&A System**: Ask questions about your financial data and get instant, data-driven answers

### 📈 **Advanced Forecasting**
- **Hybrid ML Models**: Combines Linear Regression + ARIMA for accurate predictions
- **6-Month Projections**: Revenue, expenses, and profit forecasts with confidence intervals
- **Cash Runway Analysis**: Real-time calculation of months until cash depletion
- **Scenario Planning**: Visualize different financial trajectories

### 📄 **Professional Reporting**
- **PDF Report Generation**: Export comprehensive financial reports with AI insights
- **Strategic Recommendations**: Priority-ranked action items with specific timelines
- **Data Export**: Download processed data and analysis results
- **Custom Branding**: Personalize reports with your company information

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12 or higher
- Windows/Mac/Linux
- Groq API key (free at [console.groq.com](https://console.groq.com))

### Installation

1. **Clone the repository**
```powershell
git clone https://github.com/yourusername/ai-cfo.git
cd ai-cfo
```

2. **Create virtual environment**
```powershell
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux
```

3. **Install dependencies**
```powershell
pip install -r requirements.txt
```

4. **Set up API key** (optional but recommended)
```powershell
# Option 1: Environment variable
$env:GROQ_API_KEY = "your-api-key-here"

# Option 2: Create .env file
echo "GROQ_API_KEY=your-api-key-here" > .env
```

5. **Run the dashboard**
```powershell
streamlit run app.py
```

6. **Open browser** → http://localhost:8501

---

## 💡 Usage Guide

### Web Dashboard (Recommended)

1. **Configure AI** (Optional)
   - Enter your Groq API key in the sidebar
   - Get one free at [console.groq.com](https://console.groq.com)

2. **Upload Financial Data**
   - Click "Upload Financial Data" in sidebar
   - Supported formats: CSV, Excel (.xlsx, .xls)
   - Required columns: `date`, `description`, `amount`, `category`, `vendor`

3. **Analyze Data**
   - Click "🔍 Analyze Data" button
   - Wait for processing (5-10 seconds)
   - Explore interactive visualizations

4. **Review AI Insights**
   - **AI-Powered Insights**: Quick pattern detection
   - **AI Executive Summary**: Strategic overview (with API key)
   - **CFO Recommendations**: AI-generated action items
   - **💬 Ask Questions**: Interactive Q&A about your financial data

5. **Ask Questions (New!)**
   - Navigate to the Q&A section
   - Ask natural language questions about your data
   - Get instant AI-powered answers with specific numbers
   - Review conversation history and export insights

6. **Generate Reports**
   - Click "📄 Generate PDF Report"
   - Download professional report with AI insights
   - Export processed data as CSV

### Command Line Interface

```powershell
# Run with default settings
python main.py

# Custom data file
python main.py --data path/to/your/data.csv

# With API key for AI summary
python main.py --api-key "your-api-key" --data data.csv

# Custom starting cash balance
python main.py --cash 500000 --data data.csv
```

---

## 📁 Data Format

Your financial data should be in CSV or Excel format with these columns:

```csv
date,description,amount,category,vendor
2024-01-15,Product Sale,5000,Revenue,Client A
2024-01-20,Server Costs,-200,IT,AWS
2024-01-25,Consulting Revenue,8000,Revenue,Client B
2024-02-01,Salaries,-15000,Payroll,Employees
2024-02-05,Marketing Campaign,-3000,Marketing,Google Ads
```

### Column Descriptions:
- **date**: Transaction date (YYYY-MM-DD format)
- **description**: Transaction description
- **amount**: Transaction amount (positive for revenue, negative for expenses)
- **category**: Transaction category (Revenue, IT, Marketing, Payroll, etc.)
- **vendor**: Vendor or customer name

### Sample Data
Sample files are included in the `data/` folder:
- `sample_financials.csv` - Basic example
- `mixed_business_financials.csv` - Comprehensive dataset

---

## 🏗️ Project Structure

```
ai_cfo/
├── app.py                          # Streamlit web dashboard
├── main.py                         # CLI analysis pipeline
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── AI_SETUP_GUIDE.md              # Detailed AI setup guide
│
├── modules/                        # Core analysis modules
│   ├── financial_loader.py        # Data loading & preprocessing
│   ├── financial_kpis.py          # KPI calculations
│   ├── financial_statements.py   # P&L & variance analysis
│   ├── forecast_engine.py         # ML forecasting (Hybrid + ARIMA)
│   ├── pdf_report.py              # PDF report generation
│   └── financial_qa.py            # AI Q&A system (NEW)
│
├── data/                           # Sample financial data
│   ├── sample_financials.csv
│   └── mixed_business_financials.csv
│
├── fonts/                          # PDF fonts
│   └── DejaVuSans.ttf
│
├── reports/                        # Generated reports (auto-created)
└── notebooks/                      # Jupyter notebooks (optional)
```

---

## 🤖 AI Features Explained

### Without API Key (Basic Mode)
- ✅ All data analysis features
- ✅ Charts and visualizations
- ✅ Rule-based recommendations
- ✅ PDF reports
- ❌ AI executive summary
- ❌ AI-generated recommendations
- ❌ AI Q&A system

### With API Key (AI-Powered Mode)
- ✅ Everything from Basic Mode
- ✅ **AI Executive Summary**: Strategic financial overview
- ✅ **AI Recommendations**: Context-aware advice with timelines
- ✅ **Expected Impact**: Predictions of recommendation outcomes
- ✅ **Enhanced PDF Reports**: AI insights included
- ✅ **💬 AI Q&A System**: Ask questions and get instant answers about your data

### Example AI Recommendation

```
🔴 CRITICAL - Revenue Recovery
Issue: 18.5% revenue decline with seasonal patterns detected

AI Analysis: The decline correlates with market saturation in Q2. 
Your customer acquisition cost increased 40% while conversion rates 
dropped 25%, indicating positioning issues.

Recommended Actions:
• Week 1: Conduct win-loss interviews with last 20 deals
• Week 2-3: Launch competitor positioning analysis
• Month 2: Pilot new value proposition with segment A
• Month 3: Implement pricing optimization based on findings
• Ongoing: Daily revenue dashboard monitoring with alerts

Expected Impact: 12-15% recovery within 90 days if executed, 
potential 20%+ upside with successful repositioning
```

### 💬 AI Q&A Feature (New!)

Ask natural language questions about your financial data and get instant, data-driven answers.

**Example Questions:**
```
Q: What is our cash runway and when will we run out of money?
A: Based on your current burn rate of $45,230/month and starting cash 
   of $300,000, you have approximately 6.6 months of runway remaining. 
   At current projections, cash will be depleted by June 2025. 
   
   RECOMMENDATION: Begin fundraising immediately, as the process typically 
   takes 4-6 months. Cut non-essential expenses by 20% to extend runway 
   to 8 months.

Q: Which expense categories should we prioritize cutting?
A: Analysis shows:
   1. Marketing: $18,450/month (40% of expenses) - highest category
   2. Software/Tools: $6,200/month (14%) - 15% growth month-over-month
   3. Office/Overhead: $4,100/month (9%) - flat trend
   
   PRIORITY CUTS:
   - Marketing: Reduce CAC by optimizing channel mix (save $5-7K/month)
   - Software: Audit unused subscriptions (potential $1-2K/month savings)
   - Hold office expenses flat (already optimized)
   
   Total potential savings: $6-9K/month = 2 additional months runway
```

**Features:**
- 🔍 Contextual understanding of your complete financial situation
- 📊 Specific numbers, percentages, and trends from your data
- 💡 Actionable recommendations with expected impact
- 📝 Conversation history with export capability
- ⚡ AI-generated suggested questions based on your data

**[See Complete Q&A Guide](QA_FEATURE_GUIDE.md)** for examples and best practices.

---

## 🎯 Use Cases

### For Startups
- **Cash Runway Monitoring**: Know exactly when you'll run out of money
- **Burn Rate Optimization**: AI recommendations for cost reduction
- **Fundraising Preparation**: Board-ready reports for investors

### For Small Businesses
- **Profitability Analysis**: Understand which areas make money
- **Expense Control**: Identify cost-saving opportunities
- **Growth Planning**: Data-driven expansion decisions

### For CFOs & Finance Teams
- **Executive Reporting**: Automated board-ready summaries
- **Strategic Planning**: AI-powered 30/60/90 day priorities
- **Variance Analysis**: Month-over-month change tracking

### For Investors
- **Portfolio Monitoring**: Track portfolio company health
- **Risk Assessment**: Early warning of financial issues
- **Due Diligence**: Comprehensive financial analysis

---

## 📊 Technical Stack

### Core Technologies
- **Python 3.12**: Modern Python with latest features
- **Streamlit**: Interactive web dashboard framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing

### Machine Learning
- **Scikit-learn**: Linear regression models
- **Statsmodels**: ARIMA time series forecasting
- **Hybrid Approach**: Combines ML + statistical methods

### AI Integration
- **Groq API**: Ultra-fast LLM inference
- **Llama 3.3 70B**: Advanced language model
- **JSON Structured Output**: Reliable AI responses

### Visualization
- **Matplotlib**: Publication-quality charts
- **Seaborn**: Statistical data visualization
- **Streamlit Charts**: Interactive plots

### Reporting
- **FPDF2**: PDF generation with Unicode support
- **Custom Templates**: Professional report layouts

---

## ⚙️ Configuration

### Environment Variables
```bash
GROQ_API_KEY=your-api-key-here    # Required for AI features
```

### Starting Cash Balance
Configure in the app or via CLI:
```powershell
python main.py --cash 300000  # Default: $300,000
```

### Forecast Period
Modify in code:
```python
forecast_engine = HybridForecastEngine(periods=6)  # Default: 6 months
```

---

## 🔧 Troubleshooting

### Common Issues

**"Module not found" errors**
```powershell
pip install --upgrade -r requirements.txt
```

**Streamlit won't start**
```powershell
# Use full path to Python
D:/ai_cfo/venv/Scripts/python.exe -m streamlit run app.py
```

**PDF generation fails**
- Check font file exists: `fonts/DejaVuSans.ttf`
- Verify reports directory permissions

**AI features not working**
- Verify API key is correct
- Check internet connection
- Visit [status.groq.com](https://status.groq.com) for service status

**Data upload errors**
- Ensure CSV has required columns
- Check date format (YYYY-MM-DD)
- Remove special characters from file

### Debug Mode
Enable detailed logging:
```powershell
streamlit run app.py --logger.level=debug
```

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Development Setup
```powershell
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code formatting
black .
flake8 .
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Groq** for ultra-fast LLM inference
- **Meta** for Llama models
- **Streamlit** for amazing web framework
- **Open source community** for incredible tools

---

## 📞 Support

### Documentation
- [Quick Start Guide](AI_SETUP_GUIDE.md)
- [API Documentation](https://console.groq.com/docs)

### Community
- **Issues**: [GitHub Issues](https://github.com/yourusername/ai-cfo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/ai-cfo/discussions)

### Get Help
- 📧 Email: support@example.com
- 💬 Discord: [Join our server](https://discord.gg/example)
- 🐦 Twitter: [@example](https://twitter.com/example)

---

## 🗺️ Roadmap

### Coming Soon
- [ ] Multi-currency support
- [ ] Budget vs. Actual tracking
- [ ] Custom KPI definitions
- [ ] API endpoints for integration
- [ ] Mobile-responsive dashboard
- [ ] Real-time data sync
- [ ] Team collaboration features
- [ ] Advanced ML models (Prophet, XGBoost)

### Under Consideration
- [ ] Database integration (PostgreSQL, MongoDB)
- [ ] Cloud deployment (AWS, Azure, GCP)
- [ ] Multi-tenant support
- [ ] Advanced security (SSO, 2FA)
- [ ] Slack/Teams integration
- [ ] Automated anomaly detection

---

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

---

## 📸 Screenshots

### Dashboard Overview
![Dashboard](docs/images/dashboard.png)

### AI Recommendations
![Recommendations](docs/images/recommendations.png)

### Financial Forecasts
![Forecasts](docs/images/forecasts.png)

### PDF Report
![PDF Report](docs/images/pdf-report.png)

---

<div align="center">

**Built with ❤️ by the AI CFO Team**

[Website](https://example.com) • [Documentation](https://docs.example.com) • [Blog](https://blog.example.com)

</div>
