"""SARIMAX forecaster for time series"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import warnings
from src.models.base import BaseForecaster
from src.utils.logging_config import get_logger

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


logger = get_logger(__name__)


class SARIMAXForecaster(BaseForecaster):
    """
    SARIMAX-based time series forecaster

    Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors.

    Features:
    - ARIMA order (p, d, q)
    - Seasonal order (P, D, Q, s)
    - Optional exogenous regressors (working_days)
    - Automatic parameter selection option
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = (2, 1, 2),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 12),
        include_working_days: bool = False,
        enforce_stationarity: bool = False,
        enforce_invertibility: bool = False
    ):
        """
        Initialize SARIMAX forecaster

        Args:
            order: ARIMA order (p, d, q)
                - p: autoregressive order
                - d: differencing order
                - q: moving average order
            seasonal_order: Seasonal order (P, D, Q, s)
                - P: seasonal autoregressive order
                - D: seasonal differencing order
                - Q: seasonal moving average order
                - s: seasonal period (12 for monthly data)
            include_working_days: Add working_days as exogenous regressor
            enforce_stationarity: Enforce stationarity
            enforce_invertibility: Enforce invertibility
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is not installed. Install with: pip install statsmodels")

        super().__init__(model_name='SARIMAX')

        self.order = order
        self.seasonal_order = seasonal_order
        self.include_working_days = include_working_days
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility

        self.target_col = None
        self.df_history = None
        self.has_working_days = False

    def fit(
        self,
        df_train: pd.DataFrame,
        target_col: str
    ) -> 'SARIMAXForecaster':
        """
        Train SARIMAX model

        Args:
            df_train: Training data with date and target columns
            target_col: Name of target column to forecast

        Returns:
            Self (for method chaining)
        """
        logger.info(f"Training SARIMAX model for '{target_col}'")
        logger.info(f"  Order: {self.order}, Seasonal: {self.seasonal_order}")

        self.target_col = target_col

        # Prepare data
        df = df_train.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        # Store history
        self.df_history = df.copy()

        # Get time series
        ts_data = df.set_index('date')[target_col]

        # Check for working_days regressor
        exog = None
        self.has_working_days = self.include_working_days and 'working_days' in df.columns
        if self.has_working_days:
            exog = df.set_index('date')[['working_days']]
            logger.info("  Including working_days as exogenous regressor")

        # Fit SARIMAX model
        logger.info(f"Fitting SARIMAX on {len(ts_data)} samples...")

        # Suppress convergence warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')

            model = SARIMAX(
                ts_data,
                exog=exog,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=self.enforce_stationarity,
                enforce_invertibility=self.enforce_invertibility
            )

            self.model = model.fit(disp=False, maxiter=200)

        # Calculate training metrics (in-sample predictions)
        train_pred = self.model.fittedvalues
        y_true = ts_data.values
        y_pred = train_pred.values

        # Skip first few values that may be NaN due to differencing
        valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        self.training_metrics = self._calculate_metrics(
            y_true[valid_mask], y_pred[valid_mask]
        )

        self.is_fitted = True

        logger.info(f"SARIMAX trained (MAPE: {self.training_metrics['mape']:.2f}%)")

        return self

    def predict(
        self,
        n_periods: int,
        df_history: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Generate forecasts using SARIMAX

        Args:
            n_periods: Number of periods to forecast
            df_history: Historical data (if None, uses training data)

        Returns:
            DataFrame with forecasts
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        logger.info(f"Generating {n_periods}-step SARIMAX forecast")

        # Update history if provided
        if df_history is not None:
            self.df_history = df_history.copy()
            self.df_history['date'] = pd.to_datetime(self.df_history['date'])
            self.df_history = self.df_history.sort_values('date')

        # Get last date
        last_date = self.df_history['date'].max()

        # Generate future dates
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=n_periods,
            freq='MS'
        )

        # Prepare exogenous variables for forecast period
        exog_forecast = None
        if self.has_working_days:
            working_days_values = []
            for date in future_dates:
                # Use same month from prior year
                prior_year_data = self.df_history[
                    (self.df_history['date'].dt.month == date.month) &
                    (self.df_history['date'].dt.year == date.year - 1)
                ]
                if len(prior_year_data) > 0 and 'working_days' in prior_year_data.columns:
                    working_days_values.append(prior_year_data.iloc[0]['working_days'])
                else:
                    # Use average for this month
                    month_avg = self.df_history[
                        self.df_history['date'].dt.month == date.month
                    ]['working_days'].mean() if 'working_days' in self.df_history.columns else 20
                    working_days_values.append(month_avg)

            exog_forecast = pd.DataFrame(
                {'working_days': working_days_values},
                index=future_dates
            )

        # Generate forecast
        forecast_result = self.model.get_forecast(
            steps=n_periods,
            exog=exog_forecast
        )

        predictions = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int()

        # Create output DataFrame
        df_forecast = pd.DataFrame({
            'date': future_dates,
            self.target_col: predictions.values,
            f'{self.target_col}_lower': conf_int.iloc[:, 0].values,
            f'{self.target_col}_upper': conf_int.iloc[:, 1].values
        })

        logger.info("SARIMAX forecast complete")

        return df_forecast

    def validate(
        self,
        df_val: pd.DataFrame,
        target_col: str
    ) -> Dict[str, float]:
        """
        Validate model on holdout data

        Args:
            df_val: Validation data
            target_col: Name of target column

        Returns:
            Dictionary of validation metrics
        """
        logger.info(f"Validating SARIMAX on {len(df_val)} samples")

        df = df_val.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        # Prepare exogenous variables
        exog = None
        if self.has_working_days and 'working_days' in df.columns:
            exog = df.set_index('date')[['working_days']]

        # Generate predictions for validation period
        n_periods = len(df)
        forecast_result = self.model.get_forecast(
            steps=n_periods,
            exog=exog
        )

        y_true = df[target_col].values
        y_pred = forecast_result.predicted_mean.values

        metrics = self._calculate_metrics(y_true, y_pred)

        self.validation_metrics = metrics

        logger.info(f"  MAPE: {metrics['mape']:.2f}%")
        logger.info(f"  MAE:  {metrics['mae']:,.2f}")

        return metrics
