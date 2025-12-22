"""Simple percentage-based revenue modeling"""

import pandas as pd
import numpy as np
from typing import Dict
from sklearn.linear_model import LinearRegression
from src.utils.logging_config import get_logger


logger = get_logger(__name__)


class RevenuePercentageModel:
    """
    Simple revenue percentage model

    Models the relationship between transportation revenue and total revenue
    using historical percentage ratios with trend detection.

    Approach:
    1. Calculate historical ratio: total_revenue / transportation_revenue
    2. Analyze monthly seasonality
    3. Detect linear trend over time
    4. Apply ratio (with trend adjustment) to forecasted transportation revenue
    """

    def __init__(self):
        """Initialize Revenue Percentage Model"""
        self.monthly_ratios = None
        self.mean_ratio = None
        self.std_ratio = None
        self.trend_model = None
        self.fit_date = None
        self.is_fitted = False

    def fit(self, df_historic: pd.DataFrame) -> 'RevenuePercentageModel':
        """
        Fit percentage model on historical data

        Args:
            df_historic: DataFrame with columns [date, revenue_total, total_revenue_all]

        Returns:
            Self (for method chaining)
        """
        logger.info("Fitting Revenue Percentage Model")

        # Validate input
        required_cols = ['date', 'revenue_total', 'total_revenue_all']
        missing_cols = [col for col in required_cols if col not in df_historic.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        df = df_historic.copy()
        df['date'] = pd.to_datetime(df['date'])

        # Calculate revenue ratio
        df['revenue_ratio'] = df['total_revenue_all'] / df['revenue_total']

        # Check for violations (ratio < 1.0)
        violations = df[df['revenue_ratio'] < 1.0]
        if len(violations) > 0:
            logger.warning(
                f"⚠️  Found {len(violations)} months where total_revenue < transportation_revenue"
            )
            # Clip ratios to minimum of 1.0
            df['revenue_ratio'] = df['revenue_ratio'].clip(lower=1.0)

        # Calculate monthly seasonality
        df['month'] = df['date'].dt.month
        self.monthly_ratios = df.groupby('month')['revenue_ratio'].agg(['mean', 'std', 'count'])

        logger.info("  Monthly ratio statistics:")
        for month, row in self.monthly_ratios.iterrows():
            logger.info(f"    Month {month:2d}: {row['mean']:.3f} ± {row['std']:.3f} (n={int(row['count'])})")

        # Calculate overall statistics
        self.mean_ratio = df['revenue_ratio'].mean()
        self.std_ratio = df['revenue_ratio'].std()

        logger.info(f"  Overall ratio: {self.mean_ratio:.3f} ± {self.std_ratio:.3f}")

        # Detect trend over time
        self.fit_date = df['date'].min()
        X = (df['date'] - self.fit_date).dt.days.values.reshape(-1, 1)
        y = df['revenue_ratio'].values

        self.trend_model = LinearRegression()
        self.trend_model.fit(X, y)

        trend_slope = self.trend_model.coef_[0]
        logger.info(f"  Trend slope: {trend_slope:.6f} per day ({trend_slope * 365:.4f} per year)")

        self.is_fitted = True

        logger.info("✅ Revenue Percentage Model fitted")

        return self

    def predict(self, df_forecast: pd.DataFrame) -> pd.DataFrame:
        """
        Apply revenue ratio to forecasted transportation revenue

        Args:
            df_forecast: DataFrame with columns [date, revenue_total]
                        (transportation revenue forecast)

        Returns:
            DataFrame with added columns [revenue_ratio, total_revenue_all]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        logger.info(f"Applying revenue percentage model to {len(df_forecast)} forecasts")

        df = df_forecast.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month

        # Get base ratio from monthly averages
        df['base_ratio'] = df['month'].map(self.monthly_ratios['mean'])

        # Apply trend adjustment
        days_since_fit = (df['date'] - self.fit_date).dt.days
        trend_adjustment = self.trend_model.predict(days_since_fit.values.reshape(-1, 1))

        # Final ratio = base_ratio + (trend - mean_trend)
        mean_trend = self.trend_model.predict(np.array([[0]]))[0]
        df['revenue_ratio'] = df['base_ratio'] + (trend_adjustment - mean_trend)

        # Ensure ratio >= 1.0
        df['revenue_ratio'] = df['revenue_ratio'].clip(lower=1.0)

        # Calculate total revenue
        df['total_revenue_all'] = df['revenue_total'] * df['revenue_ratio']

        # Log summary
        logger.info(f"  Ratio range: {df['revenue_ratio'].min():.3f} - {df['revenue_ratio'].max():.3f}")
        logger.info(f"  Mean ratio: {df['revenue_ratio'].mean():.3f}")
        logger.info(f"  Total revenue: CHF {df['total_revenue_all'].sum():,.0f}")

        return df

    def get_statistics(self) -> Dict:
        """
        Get model statistics

        Returns:
            Dictionary with model statistics
        """
        if not self.is_fitted:
            return {}

        return {
            'mean_ratio': float(self.mean_ratio),
            'std_ratio': float(self.std_ratio),
            'trend_slope_per_day': float(self.trend_model.coef_[0]),
            'trend_slope_per_year': float(self.trend_model.coef_[0] * 365),
            'monthly_ratios': self.monthly_ratios['mean'].to_dict()
        }

    def validate(
        self,
        df_actual: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Validate model on actual data

        Args:
            df_actual: DataFrame with columns [date, revenue_total, total_revenue_all]

        Returns:
            Dictionary of validation metrics
        """
        logger.info("Validating Revenue Percentage Model")

        # Generate predictions for actual dates
        df_pred = self.predict(df_actual[['date', 'revenue_total']])

        # Calculate metrics
        actual = df_actual['total_revenue_all'].values
        predicted = df_pred['total_revenue_all'].values

        # MAPE
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100

        # MAE
        mae = np.mean(np.abs(actual - predicted))

        # RMSE
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))

        metrics = {
            'mape': float(mape),
            'mae': float(mae),
            'rmse': float(rmse)
        }

        logger.info(f"  Validation MAPE: {mape:.2f}%")

        return metrics
