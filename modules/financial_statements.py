import pandas as pd

class FinancialStatements:

    def generate_pnl(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates a monthly Profit & Loss statement.
        """
        revenue = df[df["amount"] > 0].groupby("month")["amount"].sum()
        expenses = -df[df["amount"] < 0].groupby("month")["amount"].sum()

        pnl = pd.DataFrame({
            "revenue": revenue,
            "expenses": expenses
        })

        pnl["net_profit"] = pnl["revenue"] - pnl["expenses"]
        return pnl


    def variance_analysis(self, pnl: pd.DataFrame) -> pd.DataFrame:
        """
        Computes month-over-month % change.
        """
        variance = pnl.pct_change().fillna(0) * 100

        variance = variance.rename(columns={
            "revenue": "revenue_change_pct",
            "expenses": "expense_change_pct",
            "net_profit": "profit_change_pct"
        })

        return variance
