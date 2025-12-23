import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA


class HybridForecastEngine:
    """
    Hybrid forecasting engine (ML Regression + ARIMA)
    Fully compatible with Windows + Python 3.12 (no Prophet dependency).
    """

    def __init__(self, periods: int = 6):
        self.periods = periods

    # ------------------------------
    # PREPARATION
    # ------------------------------
    def _prepare_monthly_series(self, df: pd.DataFrame) -> pd.DataFrame:
        revenue = df[df["amount"] > 0].groupby("month")["amount"].sum()
        expenses = -df[df["amount"] < 0].groupby("month")["amount"].sum()
        net_profit = revenue - expenses

        monthly = pd.DataFrame({
            "revenue": revenue,
            "expenses": expenses,
            "net_profit": net_profit,
        }).reset_index()

        monthly["ds"] = monthly["month"].dt.to_timestamp()
        return monthly

    # ------------------------------
    # ML TREND FORECAST (Prophet alternative)
    # ------------------------------
    def _ml_forecast(self, series: pd.Series, periods: int):
        """
        Simple linear regression trend forecasting.
        Works everywhere without external dependencies.
        """
        df = series.reset_index()
        df["t"] = np.arange(len(df))  # time index
        model = LinearRegression()
        model.fit(df[["t"]], df[series.name])

        future_t = np.arange(len(df), len(df) + periods)
        yhat = model.predict(future_t.reshape(-1, 1))

        future_dates = pd.date_range(
            df["ds"].iloc[-1] + pd.offsets.MonthBegin(1),
            periods=periods,
            freq="MS"
        )

        return pd.DataFrame({
            "ds": future_dates,
            "yhat": yhat,
            "yhat_lower": yhat * 0.95,
            "yhat_upper": yhat * 1.05
        })

    # ------------------------------
    # ARIMA FORECAST
    # ------------------------------
    def _arima_forecast(self, series: pd.Series, periods: int):
        model = ARIMA(series.values, order=(1, 1, 1))
        model_fit = model.fit()
    
        forecast = model_fit.get_forecast(steps=periods)
        yhat = forecast.predicted_mean
        conf = forecast.conf_int(alpha=0.2)  # returns NumPy array
    
        future_dates = pd.date_range(
            series.index[-1] + pd.offsets.MonthBegin(1),
            periods=periods,
            freq="MS"
        )
    
        df = pd.DataFrame({
            "ds": future_dates,
            "yhat": yhat,
            "yhat_lower": conf[:, 0],
            "yhat_upper": conf[:, 1],
        })
    
        return df


    # ------------------------------
    # MAIN PUBLIC METHOD
    # ------------------------------
    def forecast_all(self, df: pd.DataFrame):
        monthly = self._prepare_monthly_series(df)
        monthly = monthly.sort_values("ds")

        revenue_ts = monthly.set_index("ds")["revenue"]
        expenses_ts = monthly.set_index("ds")["expenses"]
        profit_ts = monthly.set_index("ds")["net_profit"]

        results = {}

        for name, series in [
            ("revenue", revenue_ts),
            ("expenses", expenses_ts),
            ("net_profit", profit_ts),
        ]:

            # ML trend forecast (Prophet replacement)
            ml_fc = self._ml_forecast(series, self.periods)

            # ARIMA forecast
            arima_fc = self._arima_forecast(series, self.periods)

            # Hybrid = average of ML + ARIMA
            hybrid = ml_fc.copy()
            hybrid["yhat_arima"] = arima_fc["yhat"].values
            hybrid["yhat"] = (hybrid["yhat"] + hybrid["yhat_arima"]) / 2

            # Save results
            results[name] = {
                "history": monthly[["ds", name]].rename(columns={name: "y"}),
                "ml": ml_fc,
                "arima": arima_fc,
                "hybrid": hybrid,
            }

        return results

    # ------------------------------
    # CASH RUNWAY FORECAST
    # ------------------------------
    def compute_cash_runway(self, profit_forecast, starting_cash=300000):
        df = profit_forecast[["ds", "yhat"]].copy()
        df["cash_balance"] = starting_cash + df["yhat"].cumsum()
        df["is_negative"] = df["cash_balance"] < 0
        return df
