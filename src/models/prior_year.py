"""Prior Year forecaster - uses same month from previous year"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from src.models.base import BaseForecaster
from src.utils.logging_config import get_logger


logger = get_logger(__name__)


class PriorYearForecaster(BaseForecaster):
    """
    Prior Year forecaster

    Uses the actual value from the same month of the previous year.
    This is the traditional "human" baseline method.
    """

    def __init__(self):
        """Initialize Prior Year forecaster"""
        super().__init__(model_name='Prior Year')
        self.history = None
        self.target_col = None

    def fit(
        self,
        df_train: pd.DataFrame,
        target_col: str
    ) -> 'PriorYearForecaster':
        """
        Fit Prior Year model

        Stores historical data for lookups.

        Args:
            df_train: Training data with date and target columns
            target_col: Name of target column to forecast

        Returns:
            Self (for method chaining)
        """
        logger.debug(f"Fitting Prior Year model for '{target_col}'")

        self.target_col = target_col

        # Ensure date column
        if 'date' not in df_train.columns:
            raise ValueError("DataFrame must have 'date' column")

        df = df_train.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month

        # Store history indexed by year-month
        self.history = df.set_index(['year', 'month'])[target_col].to_dict()

        self.is_fitted = True

        # Calculate in-sample metrics (compare each month to same month prior year)
        df_sorted = df.sort_values('date')
        y_true = []
        y_pred = []

        for _, row in df_sorted.iterrows():
            prior_year_key = (row['year'] - 1, row['month'])
            if prior_year_key in self.history:
                y_true.append(row[target_col])
                y_pred.append(self.history[prior_year_key])

        if y_true:
            self.training_metrics = self._calculate_metrics(
                np.array(y_true), np.array(y_pred)
            )
            logger.debug(f"Prior Year fitted (MAPE: {self.training_metrics['mape']:.2f}%)")
        else:
            self.training_metrics = {'mape': np.nan, 'mae': np.nan, 'rmse': np.nan, 'r2': np.nan}
            logger.debug("Prior Year fitted (no prior year data for metrics)")

        return self

    def predict(
        self,
        n_periods: int,
        df_history: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Generate forecasts using prior year values

        Args:
            n_periods: Number of periods to forecast
            df_history: Historical data (uses last date to determine start)

        Returns:
            DataFrame with forecasts
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        logger.info(f"Generating {n_periods}-step Prior Year forecast")

        # Update history if provided
        if df_history is not None and 'date' in df_history.columns:
            df = df_history.copy()
            df['date'] = pd.to_datetime(df['date'])
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month

            # Update history with new data
            for _, row in df.iterrows():
                key = (row['year'], row['month'])
                if self.target_col in row:
                    self.history[key] = row[self.target_col]

            last_date = df['date'].max()
        else:
            # Use max date from history
            if self.history:
                max_year = max(k[0] for k in self.history.keys())
                max_month = max(k[1] for k in self.history.keys() if k[0] == max_year)
                last_date = pd.Timestamp(year=max_year, month=max_month, day=1)
            else:
                last_date = pd.Timestamp.now()

        # Generate forecasts
        forecasts = []
        for i in range(n_periods):
            next_date = last_date + pd.DateOffset(months=i+1)
            year = next_date.year
            month = next_date.month

            # Look up same month from prior year
            prior_year_key = (year - 1, month)
            if prior_year_key in self.history:
                forecast_value = self.history[prior_year_key]
            else:
                # Fallback: try two years ago
                two_years_ago = (year - 2, month)
                if two_years_ago in self.history:
                    forecast_value = self.history[two_years_ago]
                    logger.warning(f"  {next_date.strftime('%Y-%m')}: Using 2 years ago (no prior year)")
                else:
                    forecast_value = np.nan
                    logger.warning(f"  {next_date.strftime('%Y-%m')}: No prior year data")

            forecasts.append({
                'date': next_date,
                self.target_col: forecast_value
            })

        df_forecast = pd.DataFrame(forecasts)

        logger.info("Prior Year forecast complete")

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
        logger.info(f"Validating Prior Year on {len(df_val)} samples")

        df = df_val.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month

        y_true = []
        y_pred = []

        for _, row in df.iterrows():
            actual = row[target_col]
            prior_year_key = (row['year'] - 1, row['month'])

            if prior_year_key in self.history:
                y_true.append(actual)
                y_pred.append(self.history[prior_year_key])

        if not y_true:
            logger.warning("No prior year data available for validation")
            return {'mape': np.nan, 'mae': np.nan, 'rmse': np.nan, 'r2': np.nan}

        metrics = self._calculate_metrics(np.array(y_true), np.array(y_pred))

        self.validation_metrics = metrics

        logger.info(f"  MAPE: {metrics['mape']:.2f}%")
        logger.info(f"  MAE:  {metrics['mae']:,.2f}")

        return metrics
