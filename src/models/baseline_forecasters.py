"""Baseline forecasting models for benchmarking"""

import pandas as pd
import numpy as np
from typing import Dict
from src.models.base import BaseForecaster
from src.utils.logging_config import get_logger


logger = get_logger(__name__)


class SeasonalNaiveForecaster(BaseForecaster):
    """
    Seasonal Naive forecaster

    Uses the same month from the previous year as the forecast.
    Simple but effective baseline for seasonal data.
    """

    def __init__(self):
        """Initialize Seasonal Naive forecaster"""
        super().__init__(model_name='Seasonal Naive')
        self.monthly_values = {}
        self.target_col = None

    def fit(
        self,
        df_train: pd.DataFrame,
        target_col: str
    ) -> 'SeasonalNaiveForecaster':
        """
        Fit Seasonal Naive model

        Calculates average value for each month across all years.

        Args:
            df_train: Training data with date and target columns
            target_col: Name of target column to forecast

        Returns:
            Self (for method chaining)
        """
        logger.debug(f"Fitting Seasonal Naive model for '{target_col}'")

        self.target_col = target_col

        # Ensure date column
        if 'date' not in df_train.columns:
            raise ValueError("DataFrame must have 'date' column")

        df_train['date'] = pd.to_datetime(df_train['date'])
        df_train['month'] = df_train['date'].dt.month

        # Calculate average for each month
        self.monthly_values = df_train.groupby('month')[target_col].mean().to_dict()

        self.is_fitted = True

        # Calculate training metrics (in-sample predictions)
        y_true = df_train[target_col].values
        y_pred = df_train['month'].map(self.monthly_values).values
        self.training_metrics = self._calculate_metrics(y_true, y_pred)

        logger.debug(f"✅ Seasonal Naive fitted (MAPE: {self.training_metrics['mape']:.2f}%)")

        return self

    def predict(
        self,
        n_periods: int,
        df_history: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Generate forecasts using seasonal naive method

        Args:
            n_periods: Number of periods to forecast
            df_history: Historical data (uses last date to determine start)

        Returns:
            DataFrame with forecasts
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        logger.info(f"Generating {n_periods}-step Seasonal Naive forecast")

        # Determine start date
        if df_history is not None and 'date' in df_history.columns:
            last_date = pd.to_datetime(df_history['date']).max()
        else:
            # Default to current date
            last_date = pd.Timestamp.now()

        # Generate forecasts
        forecasts = []
        for i in range(n_periods):
            next_date = last_date + pd.DateOffset(months=i+1)
            month = next_date.month

            # Get forecast from monthly average
            forecast_value = self.monthly_values.get(month, np.nan)

            forecasts.append({
                'date': next_date,
                self.target_col: forecast_value
            })

        df_forecast = pd.DataFrame(forecasts)

        logger.info(f"✅ Seasonal Naive forecast complete")

        return df_forecast


class MovingAverageForecaster(BaseForecaster):
    """
    Moving Average forecaster

    Uses the average of the last N periods as the forecast.
    """

    def __init__(self, window: int = 3):
        """
        Initialize Moving Average forecaster

        Args:
            window: Number of periods to average (3 or 6 months typical)
        """
        super().__init__(model_name=f'MA-{window}')
        self.window = window
        self.target_col = None
        self.last_values = None

    def fit(
        self,
        df_train: pd.DataFrame,
        target_col: str
    ) -> 'MovingAverageForecaster':
        """
        Fit Moving Average model

        Stores the last N values for forecasting.

        Args:
            df_train: Training data with date and target columns
            target_col: Name of target column to forecast

        Returns:
            Self (for method chaining)
        """
        logger.debug(f"Fitting Moving Average (window={self.window}) for '{target_col}'")

        self.target_col = target_col

        # Ensure sorted by date
        df_sorted = df_train.sort_values('date')

        # Get last N values
        self.last_values = df_sorted[target_col].tail(self.window).values

        self.is_fitted = True

        # Calculate simple training metrics (constant prediction = mean of window)
        forecast_value = np.mean(self.last_values)
        y_true = df_sorted[target_col].tail(self.window).values
        y_pred = np.full_like(y_true, forecast_value)
        self.training_metrics = self._calculate_metrics(y_true, y_pred)

        logger.debug(f"✅ Moving Average fitted (MAPE: {self.training_metrics['mape']:.2f}%)")

        return self

    def predict(
        self,
        n_periods: int,
        df_history: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Generate forecasts using moving average

        Args:
            n_periods: Number of periods to forecast
            df_history: Historical data (uses last N values if provided)

        Returns:
            DataFrame with forecasts
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        logger.info(f"Generating {n_periods}-step Moving Average forecast")

        # Update last values if history provided
        if df_history is not None and self.target_col in df_history.columns:
            df_sorted = df_history.sort_values('date')
            self.last_values = df_sorted[self.target_col].tail(self.window).values

        # Determine start date
        if df_history is not None and 'date' in df_history.columns:
            last_date = pd.to_datetime(df_history['date']).max()
        else:
            last_date = pd.Timestamp.now()

        # Calculate moving average
        forecast_value = np.mean(self.last_values)

        # Generate forecasts (constant value)
        forecasts = []
        for i in range(n_periods):
            next_date = last_date + pd.DateOffset(months=i+1)

            forecasts.append({
                'date': next_date,
                self.target_col: forecast_value
            })

        df_forecast = pd.DataFrame(forecasts)

        logger.info(f"✅ Moving Average forecast complete (value: {forecast_value:,.0f})")

        return df_forecast


class LinearTrendForecaster(BaseForecaster):
    """
    Linear Trend forecaster

    Fits a linear regression model to the time series and extrapolates.
    """

    def __init__(self):
        """Initialize Linear Trend forecaster"""
        super().__init__(model_name='Linear Trend')
        self.slope = None
        self.intercept = None
        self.target_col = None
        self.last_date = None
        self.first_date = None

    def fit(
        self,
        df_train: pd.DataFrame,
        target_col: str
    ) -> 'LinearTrendForecaster':
        """
        Fit Linear Trend model

        Fits linear regression: y = slope * time + intercept

        Args:
            df_train: Training data with date and target columns
            target_col: Name of target column to forecast

        Returns:
            Self (for method chaining)
        """
        logger.debug(f"Fitting Linear Trend model for '{target_col}'")

        self.target_col = target_col

        # Ensure sorted by date
        df_sorted = df_train.sort_values('date')

        # Convert dates to numeric (days since first date)
        self.first_date = df_sorted['date'].min()
        self.last_date = df_sorted['date'].max()

        X = (df_sorted['date'] - self.first_date).dt.days.values.reshape(-1, 1)
        y = df_sorted[target_col].values

        # Fit linear regression
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)

        self.slope = model.coef_[0]
        self.intercept = model.intercept_

        self.is_fitted = True

        # Calculate training metrics
        y_pred = model.predict(X)
        self.training_metrics = self._calculate_metrics(y, y_pred)

        logger.debug(f"✅ Linear Trend fitted (MAPE: {self.training_metrics['mape']:.2f}%)")

        return self

    def predict(
        self,
        n_periods: int,
        df_history: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Generate forecasts using linear trend extrapolation

        Args:
            n_periods: Number of periods to forecast
            df_history: Historical data (uses last date if provided)

        Returns:
            DataFrame with forecasts
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        logger.info(f"Generating {n_periods}-step Linear Trend forecast")

        # Determine start date
        if df_history is not None and 'date' in df_history.columns:
            last_date = pd.to_datetime(df_history['date']).max()
        else:
            last_date = self.last_date

        # Generate forecasts
        forecasts = []
        for i in range(n_periods):
            next_date = last_date + pd.DateOffset(months=i+1)

            # Calculate days since first date
            days_since_start = (next_date - self.first_date).days

            # Linear extrapolation
            forecast_value = self.slope * days_since_start + self.intercept

            forecasts.append({
                'date': next_date,
                self.target_col: forecast_value
            })

        df_forecast = pd.DataFrame(forecasts)

        logger.info(f"✅ Linear Trend forecast complete")

        return df_forecast
