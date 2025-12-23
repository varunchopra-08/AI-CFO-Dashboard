# modules/pdf_report.py
from fpdf import FPDF
import datetime
import pandas as pd
import re
import unicodedata
import os


FONT_PATH = os.path.join(os.path.dirname(__file__), "../fonts/DejaVuSans.ttf")

class CFOReportPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.add_font("DejaVu", "", FONT_PATH, uni=True)

    def header(self):
        self.set_font("DejaVu", "", 14)
        self.cell(0, 10, "AI CFO Financial Report", ln=True, align="C")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("DejaVu", "", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")



def add_table(pdf: CFOReportPDF, title: str, df: pd.DataFrame, col_widths=None, max_rows=12):
    pdf.set_font("DejaVu", "", 10)

    pdf.cell(0, 8, title, ln=True)
    pdf.set_font("DejaVu", "", 8)


    if df.empty:
        pdf.cell(0, 5, "No data available.", ln=True)
        pdf.ln(3)
        return

    df = df.head(max_rows)

    # Calculate available width (page width - margins)
    # Default margins are 10mm on each side
    available_width = pdf.w - 2 * pdf.l_margin
    
    if col_widths is None:
        # Distribute width evenly across columns with minimum width
        num_cols = len(df.columns)
        col_width = max(25, available_width / num_cols)  # Minimum 25mm per column
        col_widths = [col_width] * num_cols
    
    # Ensure total width doesn't exceed available width
    total_width = sum(col_widths)
    if total_width > available_width:
        scale = available_width / total_width
        col_widths = [w * scale for w in col_widths]

    # Header
    for i, col in enumerate(df.columns):
        pdf.cell(col_widths[i], 6, str(col)[:20], border=1)
    pdf.ln()

    # Rows
    for _, row in df.iterrows():
        for i, col in enumerate(df.columns):
            val = row[col]
            # Format numeric values
            if pd.api.types.is_numeric_dtype(type(val)) and not pd.isna(val):
                if abs(val) >= 1000:
                    txt = f"{val:,.0f}"  # Format with commas, no decimals
                elif abs(val) >= 1:
                    txt = f"{val:.2f}"  # Two decimals for small numbers
                else:
                    txt = str(val)
            else:
                txt = str(val)
            
            # Truncate if too long
            txt = txt[:25]
            pdf.cell(col_widths[i], 6, txt, border=1)
        pdf.ln()
    pdf.ln(4)


def generate_cfo_pdf_report(
    file_path: str,
    llm_summary: str,
    kpis: pd.DataFrame,
    pnl: pd.DataFrame,
    variance: pd.DataFrame,
    runway: pd.DataFrame,
    recommendations: list = None,
):
    pdf = CFOReportPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("DejaVu", "", 10)
    today = datetime.date.today().isoformat()
    pdf.cell(0, 8, f"Generated on: {today}", ln=True)
    pdf.ln(3)

    pdf.set_font("DejaVu", "", 12)
    pdf.cell(0, 8, "CFO Executive Summary", ln=True)

    pdf.set_font("DejaVu", "", 10)
    summary_clean = clean_text_for_pdf(llm_summary)

    # Calculate available width for multi_cell
    cell_width = pdf.w - 2 * pdf.l_margin
    
    for line in safe_split_text(summary_clean):
        if line.strip():  # Only process non-empty lines
            pdf.multi_cell(cell_width, 5, line)
        else:
            pdf.ln(5)  # Add spacing for empty lines

    pdf.ln(5)

    # Add recommendations section if provided
    if recommendations:
        pdf.set_font("DejaVu", "", 12)
        pdf.cell(0, 8, "Strategic Recommendations", ln=True)
        pdf.ln(2)
        
        pdf.set_font("DejaVu", "", 10)
        for rec in recommendations:
            # Priority and category
            priority_text = rec.get('priority', 'INFO').replace('🔴', '[CRITICAL]').replace('🟡', '[MEDIUM]').replace('🟢', '[GOOD]').replace('ℹ️', '[INFO]')
            pdf.set_font("DejaVu", "", 10)
            pdf.multi_cell(cell_width, 5, f"{priority_text} - {rec['category']}: {rec['issue']}")
            
            # Actions
            pdf.set_font("DejaVu", "", 9)
            for action in rec.get('actions', []):
                action_clean = clean_text_for_pdf(action)
                pdf.multi_cell(cell_width, 5, f"  - {action_clean}")
            pdf.ln(2)
        
        pdf.ln(3)

    # Prepare data for tables - format and limit columns
    kpis_table = kpis.reset_index()
    # Convert month to string if it exists
    if 'month' in kpis_table.columns:
        kpis_table['month'] = kpis_table['month'].astype(str).str[:10]
    
    pnl_table = pnl.reset_index()
    if 'month' in pnl_table.columns:
        pnl_table['month'] = pnl_table['month'].astype(str).str[:10]
    
    variance_table = variance.reset_index()
    if 'month' in variance_table.columns:
        variance_table['month'] = variance_table['month'].astype(str).str[:10]

    # Add tables
    add_table(pdf, "Monthly KPIs", kpis_table)
    add_table(pdf, "Profit & Loss", pnl_table)
    add_table(pdf, "Variance Analysis", variance_table)

    runway_table = runway[["ds", "cash_balance"]].copy()
    runway_table["ds"] = runway_table["ds"].dt.strftime('%Y-%m-%d')
    runway_table.columns = ["Date", "Cash Balance"]
    add_table(pdf, "Cash Runway Projection", runway_table)

    pdf.output(file_path)
    return file_path
    
def safe_split_text(text, max_len=80):
    """
    Splits text into smaller chunks so FPDF can always render safely.
    Includes aggressive fallback for extremely long tokens.
    """
    safe_lines = []
    text = clean_text_for_pdf(text)

    for line in text.split("\n"):

        # If line is blank, keep it
        if len(line.strip()) == 0:
            safe_lines.append("")
            continue

        # If normal length, keep as is
        if len(line) <= max_len:
            safe_lines.append(line)
            continue

        # Aggressive wrapping: break even mid-word
        for i in range(0, len(line), max_len):
            safe_lines.append(line[i:i+max_len])

    return safe_lines


def clean_text_for_pdf(text):
    """
    Cleans unicode, removes invisible characters, 
    normalizes dashes, and prevents long-token crashes.
    """

    # Remove zero-width and invisible Unicode characters
    text = re.sub(r"[\u200B-\u200F\u202A-\u202E]", "", text)

    # Normalize dashes
    text = text.replace("-", "-").replace("–", "-").replace("—", "-")

    # Replace tabs with spaces
    text = text.replace("\t", " ")

    # Remove markdown separators like ------
    text = re.sub(r"[-_=*]{8,}", "", text)

    # Normalize to NFKD to eliminate weird combined accents
    text = unicodedata.normalize("NFKD", text)

    return text
