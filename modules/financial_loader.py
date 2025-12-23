import pandas as pd

class FinancialLoader:
    REQUIRED_COLUMNS = ["date", "description", "amount", "category", "vendor"]

    def load_csv(self, path: str) -> pd.DataFrame:
        """
        Loads & cleans financial CSV for CFO AI engine.
        """

        df = pd.read_csv(path)

        # Normalize column names
        df.columns = df.columns.str.lower().str.strip()

        # Validate columns
        for col in self.REQUIRED_COLUMNS:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Convert date column
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])  # remove rows with bad dates

        # Convert amount to numeric
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        df = df.dropna(subset=["amount"])

        # Clean category formatting
        df["category"] = df["category"].str.strip().str.title()

        # Determine type: revenue or expense
        df["type"] = df["amount"].apply(lambda x: "Revenue" if x > 0 else "Expense")

        # Extract month/year for grouping
        df["month"] = df["date"].dt.to_period("M")
        df["year"] = df["date"].dt.year

        return df
