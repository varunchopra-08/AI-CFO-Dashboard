import pandas as pd

class FinancialKPIs:

    def monthly_kpis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes monthly KPIs:
        - Revenue
        - Expenses
        - Net profit
        - Burn rate
        - Estimated cash runway
        """

        grouped = df.groupby("month")

        # Revenue = sum of positive amounts
        revenue = grouped.apply(lambda x: x[x["amount"] > 0]["amount"].sum())

        # Expenses = sum of absolute negative amounts
        expenses = grouped.apply(lambda x: -x[x["amount"] < 0]["amount"].sum())

        net_profit = revenue - expenses

        kpi_df = pd.DataFrame({
            "revenue": revenue,
            "expenses": expenses,
            "net_profit": net_profit
        })

        # Burn rate = only when profit is negative
        kpi_df["burn_rate"] = kpi_df["net_profit"].apply(lambda x: -x if x < 0 else 0)

        # Simple estimated cash runway
        starting_cash = 300000  
        avg_burn = kpi_df["burn_rate"].replace(0, pd.NA).mean()

        if pd.notna(avg_burn) and avg_burn > 0:
            runway_months = starting_cash / avg_burn
        else:
            runway_months = None

        kpi_df["estimated_runway_months"] = runway_months

        return kpi_df
