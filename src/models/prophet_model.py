"""Prophet forecaster for time series"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from src.models.base import BaseForecaster
from src.utils.logging_config import get_logger

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


logger = get_logger(__name__)


class ProphetForecaster(BaseForecaster):
    """
    Prophet-based time series forecaster

    Features:
    - Automatic seasonality detection
    - Custom seasonalities (quarterly, monthly)
    - Regressors support (working_days)
    - Uncertainty intervals
    """

    def __init__(
        self,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = False,
        daily_seasonality: bool = False,
        seasonality_mode: str = 'multiplicative',
        changepoint_prior_scale: float = 0.05,
        include_working_days: bool = False
    ):
        """
        Initialize Prophet forecaster

        Args:
            yearly_seasonality: Include yearly seasonality
            weekly_seasonality: Include weekly seasonality (False for monthly data)
            daily_seasonality: Include daily seasonality (False for monthly data)
            seasonality_mode: 'multiplicative' or 'additive'
            changepoint_prior_scale: Flexibility of trend changepoints
            include_working_days: Add working_days as regressor
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is not installed. Install with: pip install prophet")

        super().__init__(model_name='Prophet')

        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.seasonality_mode = seasonality_mode
        self.changepoint_prior_scale = changepoint_prior_scale
        self.include_working_days = include_working_days

        self.target_col = None
        self.df_history = None
        self.has_working_days = False

    def fit(
        self,
        df_train: pd.DataFrame,
        target_col: str
    ) -> 'ProphetForecaster':
        """
        Train Prophet model

        Args:
            df_train: Training data with date and target columns
            target_col: Name of target column to forecast

        Returns:
            Self (for method chaining)
        """
        logger.info(f"Training Prophet model for '{target_col}'")

        self.target_col = target_col

        # Prepare data in Prophet format
        df = df_train.copy()
        df['date'] = pd.to_datetime(df['date'])

        prophet_df = pd.DataFrame({
            'ds': df['date'],
            'y': df[target_col]
        })

        # Check for working_days regressor
        self.has_working_days = self.include_working_days and 'working_days' in df.columns
        if self.has_working_days:
            prophet_df['working_days'] = df['working_days'].values
            logger.info("  Including working_days as regressor")

        # Store history for forecasting
        self.df_history = df.copy()

        # Initialize Prophet model
        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            seasonality_mode=self.seasonality_mode,
            changepoint_prior_scale=self.changepoint_prior_scale
        )

        # Add custom seasonalities for monthly data
        self.model.add_seasonality(name='quarterly', period=91.25, fourier_order=5)
        self.model.add_seasonality(name='monthly', period=30.5, fourier_order=10)

        # Add working_days regressor if available
        if self.has_working_days:
            self.model.add_regressor('working_days')

        # Fit model (suppress Prophet's verbose output)
        import logging
        prophet_logger = logging.getLogger('prophet')
        prophet_logger.setLevel(logging.WARNING)
        cmdstanpy_logger = logging.getLogger('cmdstanpy')
        cmdstanpy_logger.setLevel(logging.WARNING)

        logger.info(f"Fitting Prophet on {len(prophet_df)} samples...")
        self.model.fit(prophet_df)

        # Calculate training metrics
        train_pred = self.model.predict(prophet_df)
        y_true = prophet_df['y'].values
        y_pred = train_pred['yhat'].values

        self.training_metrics = self._calculate_metrics(y_true, y_pred)

        self.is_fitted = True

        logger.info(f"Prophet trained (MAPE: {self.training_metrics['mape']:.2f}%)")

        return self

    def predict(
        self,
        n_periods: int,
        df_history: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Generate forecasts using Prophet

        Args:
            n_periods: Number of periods to forecast
            df_history: Historical data (if None, uses training data)

        Returns:
            DataFrame with forecasts
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        logger.info(f"Generating {n_periods}-step Prophet forecast")

        # Update history if provided
        if df_history is not None:
            self.df_history = df_history.copy()
            self.df_history['date'] = pd.to_datetime(self.df_history['date'])

        # Get last date
        last_date = self.df_history['date'].max()

        # Create future dataframe
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=n_periods,
            freq='MS'
        )

        future_df = pd.DataFrame({'ds': future_dates})

        # Add working_days if model uses it
        if self.has_working_days:
            # Use same month from prior year for working_days
            working_days_values = []
            for date in future_dates:
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

            future_df['working_days'] = working_days_values

        # Generate predictions
        forecast = self.model.predict(future_df)

        # Create output DataFrame
        df_forecast = pd.DataFrame({
            'date': forecast['ds'],
            self.target_col: forecast['yhat'],
            f'{self.target_col}_lower': forecast['yhat_lower'],
            f'{self.target_col}_upper': forecast['yhat_upper']
        })

        logger.info("Prophet forecast complete")

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
        logger.info(f"Validating Prophet on {len(df_val)} samples")

        df = df_val.copy()
        df['date'] = pd.to_datetime(df['date'])

        # Prepare Prophet format
        prophet_df = pd.DataFrame({
            'ds': df['date'],
            'y': df[target_col]
        })

        if self.has_working_days and 'working_days' in df.columns:
            prophet_df['working_days'] = df['working_days'].values

        # Generate predictions for validation period
        forecast = self.model.predict(prophet_df)

        y_true = prophet_df['y'].values
        y_pred = forecast['yhat'].values

        metrics = self._calculate_metrics(y_true, y_pred)

        self.validation_metrics = metrics

        logger.info(f"  MAPE: {metrics['mape']:.2f}%")
        logger.info(f"  MAE:  {metrics['mae']:,.2f}")

        return metrics
