"""ML-based revenue ratio modeling"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import xgboost as xgb
from src.utils.config import ConfigLoader
from src.utils.logging_config import get_logger


logger = get_logger(__name__)


class RevenueMLModel:
    """
    ML-based revenue ratio model

    Uses XGBoost to predict the revenue ratio (total / transportation)
    based on temporal and business features.

    Features:
    - Temporal: month, quarter, year
    - Business: transportation_revenue, external_drivers, personnel_costs
    - Lags: revenue_ratio lag 1, 3, 6 months
    - Trends: 3-month moving average of ratio
    """

    def __init__(self, config: Optional[ConfigLoader] = None):
        """
        Initialize ML Revenue Model

        Args:
            config: Configuration loader instance
        """
        self.config = config if config else ConfigLoader()

        # XGBoost parameters (use same as main forecasting)
        xgb_config = self.config.get('models.xgboost', {})
        self.n_estimators = xgb_config.get('n_estimators', 200)
        self.max_depth = xgb_config.get('max_depth', 6)
        self.learning_rate = xgb_config.get('learning_rate', 0.05)
        self.random_state = xgb_config.get('random_state', 42)

        self.model = None
        self.feature_cols = None
        self.is_fitted = False
        self.validation_mape = None

    def fit(self, df_historic: pd.DataFrame) -> 'RevenueMLModel':
        """
        Train ML model to predict revenue ratio

        Args:
            df_historic: DataFrame with columns [date, month, quarter, year,
                        revenue_total, total_revenue_all, external_drivers,
                        personnel_costs (optional)]

        Returns:
            Self (for method chaining)
        """
        logger.info("Fitting Revenue ML Model")

        # Validate input
        required_cols = ['date', 'revenue_total', 'total_revenue_all']
        missing_cols = [col for col in required_cols if col not in df_historic.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        df = df_historic.copy()
        df['date'] = pd.to_datetime(df['date'])

        # Calculate revenue ratio (target variable)
        df['revenue_ratio'] = df['total_revenue_all'] / df['revenue_total']

        # Clip ratios to minimum of 1.0
        df['revenue_ratio'] = df['revenue_ratio'].clip(lower=1.0)

        # Create features
        df = self._create_features(df)

        # Define feature columns
        self.feature_cols = [
            'month', 'quarter', 'year',
            'revenue_total', 'external_drivers'
        ]

        # Add personnel costs if available
        if 'personnel_costs' in df.columns:
            self.feature_cols.append('personnel_costs')
            logger.info("  ✓ Including personnel_costs as feature")

        # Add lag features
        lag_features = [col for col in df.columns if col.startswith('revenue_ratio_lag_')]
        self.feature_cols.extend(lag_features)

        # Add rolling features
        rolling_features = [col for col in df.columns if col.startswith('revenue_ratio_rolling_')]
        self.feature_cols.extend(rolling_features)

        logger.info(f"  Features: {len(self.feature_cols)}")

        # Remove rows with NaN (first N months with incomplete lags)
        df_clean = df.dropna(subset=self.feature_cols + ['revenue_ratio'])

        # Train/validation split (last 6 months for validation)
        split_idx = len(df_clean) - 6

        if split_idx < 12:  # Need at least 12 months for training
            logger.warning("⚠️  Insufficient data for train/val split, using all data for training")
            X_train = df_clean[self.feature_cols]
            y_train = df_clean['revenue_ratio']
            X_val = None
            y_val = None
        else:
            X_train = df_clean.iloc[:split_idx][self.feature_cols]
            X_val = df_clean.iloc[split_idx:][self.feature_cols]
            y_train = df_clean.iloc[:split_idx]['revenue_ratio']
            y_val = df_clean.iloc[split_idx:]['revenue_ratio']

        logger.info(f"  Training samples: {len(X_train)}")
        if X_val is not None:
            logger.info(f"  Validation samples: {len(X_val)}")

        # Train model
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            objective='reg:squarederror',
            verbosity=0
        )

        self.model.fit(X_train, y_train)

        # Calculate validation metrics
        if X_val is not None and y_val is not None:
            val_predictions = self.model.predict(X_val)
            self.validation_mape = np.mean(np.abs((y_val - val_predictions) / y_val)) * 100
            logger.info(f"  Validation MAPE: {self.validation_mape:.2f}%")
        else:
            self.validation_mape = None

        self.is_fitted = True

        logger.info("✅ Revenue ML Model fitted")

        return self

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for ML model

        Args:
            df: DataFrame with basic columns

        Returns:
            DataFrame with features added
        """
        df = df.copy()

        # Ensure temporal features exist
        if 'month' not in df.columns:
            df['month'] = df['date'].dt.month
        if 'quarter' not in df.columns:
            df['quarter'] = df['date'].dt.quarter
        if 'year' not in df.columns:
            df['year'] = df['date'].dt.year

        # Create lag features for revenue_ratio
        for lag in [1, 3, 6]:
            df[f'revenue_ratio_lag_{lag}'] = df['revenue_ratio'].shift(lag)

        # Create rolling features for revenue_ratio
        for window in [3, 6]:
            df[f'revenue_ratio_rolling_mean_{window}'] = (
                df['revenue_ratio'].rolling(window=window, min_periods=1).mean()
            )

        return df

    def predict(self, df_forecast: pd.DataFrame) -> pd.DataFrame:
        """
        Predict revenue ratio and calculate total revenue

        Args:
            df_forecast: DataFrame with columns [date, revenue_total, external_drivers,
                        personnel_costs (optional)]

        Returns:
            DataFrame with added columns [revenue_ratio, total_revenue_all]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        logger.info(f"Applying Revenue ML Model to {len(df_forecast)} forecasts")

        df = df_forecast.copy()
        df['date'] = pd.to_datetime(df['date'])

        # Create features
        df = self._create_features(df)

        # Handle missing lag features for forecast period
        # Use historical average or forward fill
        for col in self.feature_cols:
            if col.startswith('revenue_ratio_'):
                if df[col].isna().any():
                    # Forward fill or use mean ratio
                    df[col] = df[col].fillna(method='ffill').fillna(1.1)  # Default to 1.1 if all missing

        # Make predictions
        X_pred = df[self.feature_cols]

        # Handle any remaining missing values
        X_pred = X_pred.fillna(0)

        df['revenue_ratio'] = self.model.predict(X_pred)

        # Ensure ratio >= 1.0
        df['revenue_ratio'] = df['revenue_ratio'].clip(lower=1.0)

        # Calculate total revenue
        df['total_revenue_all'] = df['revenue_total'] * df['revenue_ratio']

        # Log summary
        logger.info(f"  Ratio range: {df['revenue_ratio'].min():.3f} - {df['revenue_ratio'].max():.3f}")
        logger.info(f"  Mean ratio: {df['revenue_ratio'].mean():.3f}")
        logger.info(f"  Total revenue: CHF {df['total_revenue_all'].sum():,.0f}")

        return df

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores

        Returns:
            DataFrame with features and importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importance")

        importance = self.model.feature_importances_
        df_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': importance
        }).sort_values('importance', ascending=False)

        return df_importance

    def validate(
        self,
        df_actual: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Validate model on actual data

        Args:
            df_actual: DataFrame with columns [date, revenue_total, total_revenue_all, ...]

        Returns:
            Dictionary of validation metrics
        """
        logger.info("Validating Revenue ML Model")

        # Generate predictions for actual dates
        df_pred = self.predict(df_actual[['date', 'revenue_total', 'external_drivers'] +
                                         (['personnel_costs'] if 'personnel_costs' in df_actual.columns else [])])

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
