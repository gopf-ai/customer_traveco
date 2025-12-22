"""XGBoost forecaster for time series"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from src.models.base import BaseForecaster
from src.features.engineering import TravecomFeatureEngine
from src.utils.config import ConfigLoader
from src.utils.logging_config import get_logger

# Rich components for progress tracking
try:
    from rich.console import Console
    from rich.status import Status
    RICH_AVAILABLE = True
    rich_console = Console()
except ImportError:
    RICH_AVAILABLE = False
    rich_console = None


logger = get_logger(__name__)


class XGBoostForecaster(BaseForecaster):
    """
    XGBoost-based time series forecaster

    Features:
    - Gradient boosting with XGBoost
    - Lag features (1, 3, 6, 12 months)
    - Rolling statistics
    - Temporal features
    - Recursive multi-step forecasting
    - Data leakage prevention
    """

    def __init__(
        self,
        config: Optional[ConfigLoader] = None,
        validation_months: int = 6,
        cv_folds: Optional[int] = None
    ):
        """
        Initialize XGBoost forecaster

        Args:
            config: Configuration loader instance
            validation_months: Number of months to use as validation holdout (default: 6)
            cv_folds: If specified, use time series cross-validation with N folds instead of simple holdout
        """
        super().__init__(model_name='XGBoost')

        self.config = config if config else ConfigLoader()
        self.feature_engine = TravecomFeatureEngine(self.config)

        # Validation configuration
        self.validation_months = validation_months
        self.cv_folds = cv_folds

        # XGBoost hyperparameters from config
        xgb_config = self.config.get('models.xgboost', {})
        self.n_estimators = xgb_config.get('n_estimators', 200)
        self.max_depth = xgb_config.get('max_depth', 6)
        self.learning_rate = xgb_config.get('learning_rate', 0.05)
        self.subsample = xgb_config.get('subsample', 0.8)
        self.colsample_bytree = xgb_config.get('colsample_bytree', 0.8)
        self.random_state = xgb_config.get('random_state', 42)

        self.target_col = None
        self.feature_cols = None
        self.df_full = None  # Store full dataset for recursive forecasting
        self.validation_metrics = None  # Store validation metrics

    def fit(
        self,
        df_train: pd.DataFrame,
        target_col: str
    ) -> 'XGBoostForecaster':
        """
        Train XGBoost model

        Args:
            df_train: Training data with date and target columns
            target_col: Name of target column to forecast

        Returns:
            Self (for method chaining)
        """
        logger.info(f"Training XGBoost model for '{target_col}'")

        self.target_col = target_col

        # Create features
        logger.info("Creating features...")
        df_features = self.feature_engine.create_all_features(
            df_train,
            target_col=target_col,
            date_col='date'
        )

        # Get feature columns (exclude targets and identifiers)
        self.feature_cols = self.feature_engine.get_feature_columns(df_features, target_col)

        # Remove rows with NaN in features (first N months with incomplete lags)
        df_clean = df_features.dropna(subset=self.feature_cols + [target_col])
        logger.info(f"Available samples: {len(df_clean)} (removed {len(df_features) - len(df_clean)} with missing features)")

        # Initialize model
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            objective='reg:squarederror',
            verbosity=0
        )

        # Choose validation strategy
        if self.cv_folds is not None:
            # Time Series Cross-Validation
            logger.info(f"Using {self.cv_folds}-fold Time Series Cross-Validation")
            self.validation_metrics = self._fit_with_cv(df_clean, target_col)
        else:
            # Simple holdout validation
            logger.info(f"Using holdout validation (last {self.validation_months} months)")
            self.validation_metrics = self._fit_with_holdout(df_clean, target_col)

        self.is_fitted = True

        logger.info(f"✅ Training complete")
        logger.info(f"  Validation MAPE: {self.validation_metrics['mape']:.2f}%")
        if 'mape_std' in self.validation_metrics:
            logger.info(f"  CV MAPE Std:     {self.validation_metrics['mape_std']:.2f}%")

        # Store full dataset for recursive forecasting
        self.df_full = df_features.copy()

        return self

    def _fit_with_holdout(
        self,
        df_clean: pd.DataFrame,
        target_col: str
    ) -> Dict[str, float]:
        """
        Train with simple holdout validation

        Args:
            df_clean: Cleaned data with features
            target_col: Target column name

        Returns:
            Validation metrics dictionary
        """
        # Split data: last N months for validation
        split_idx = max(0, len(df_clean) - self.validation_months)

        if split_idx == 0:
            logger.warning(f"Not enough data for {self.validation_months}-month validation. Using all data for training.")
            X_train = df_clean[self.feature_cols]
            y_train = df_clean[target_col]
            X_val = X_train
            y_val = y_train
        else:
            df_train = df_clean.iloc[:split_idx]
            df_val = df_clean.iloc[split_idx:]

            X_train = df_train[self.feature_cols]
            y_train = df_train[target_col]
            X_val = df_val[self.feature_cols]
            y_val = df_val[target_col]

            logger.info(f"  Training samples: {len(X_train)}")
            logger.info(f"  Validation samples: {len(X_val)}")

        # Train with status spinner if rich available
        if RICH_AVAILABLE and rich_console:
            with Status(
                f"[yellow]Training XGBoost on {len(X_train)} samples, {len(self.feature_cols)} features...",
                console=rich_console,
                spinner="dots"
            ):
                self.model.fit(X_train, y_train)
        else:
            logger.info(f"Training with {len(self.feature_cols)} features...")
            self.model.fit(X_train, y_train)

        # Calculate validation metrics
        val_pred = self.model.predict(X_val)
        validation_metrics = self._calculate_metrics(y_val.values, val_pred)

        return validation_metrics

    def _fit_with_cv(
        self,
        df_clean: pd.DataFrame,
        target_col: str
    ) -> Dict[str, float]:
        """
        Train with time series cross-validation

        Args:
            df_clean: Cleaned data with features
            target_col: Target column name

        Returns:
            Averaged validation metrics with std
        """
        # Prepare data
        X = df_clean[self.feature_cols]
        y = df_clean[target_col]

        # Time series split with expanding window
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)

        logger.info(f"Running {self.cv_folds}-fold time series cross-validation...")

        fold_metrics = []

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]

            logger.info(f"  Fold {fold_idx + 1}/{self.cv_folds}: Train={len(train_idx)}, Val={len(val_idx)}")

            # Train model on fold
            fold_model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                random_state=self.random_state,
                objective='reg:squarederror',
                verbosity=0
            )

            fold_model.fit(X_train_fold, y_train_fold)

            # Validate
            val_pred_fold = fold_model.predict(X_val_fold)
            fold_metric = self._calculate_metrics(y_val_fold.values, val_pred_fold)

            fold_metrics.append(fold_metric)
            logger.info(f"    Fold {fold_idx + 1} MAPE: {fold_metric['mape']:.2f}%")

        # Calculate mean and std of metrics across folds
        mean_mape = np.mean([m['mape'] for m in fold_metrics])
        std_mape = np.std([m['mape'] for m in fold_metrics])
        mean_mae = np.mean([m['mae'] for m in fold_metrics])
        mean_rmse = np.mean([m['rmse'] for m in fold_metrics])
        mean_r2 = np.mean([m['r2'] for m in fold_metrics])

        logger.info(f"  CV Results: MAPE = {mean_mape:.2f}% ± {std_mape:.2f}%")

        # Train final model on all data
        logger.info(f"Training final model on all {len(X)} samples...")
        self.model.fit(X, y)

        return {
            'mape': mean_mape,
            'mape_std': std_mape,
            'mae': mean_mae,
            'rmse': mean_rmse,
            'r2': mean_r2
        }

    def predict(
        self,
        n_periods: int,
        df_history: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Generate recursive multi-step forecasts

        Uses previous predictions as lag features for future predictions.

        Args:
            n_periods: Number of periods to forecast
            df_history: Historical data (if None, uses training data)

        Returns:
            DataFrame with forecasts
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        logger.info(f"Generating {n_periods}-step forecast for '{self.target_col}'")

        # Use stored full dataset if history not provided
        if df_history is None:
            if self.df_full is None:
                raise ValueError("No historical data available for forecasting")
            df_extended = self.df_full.copy()
        else:
            # Create features for provided history
            df_extended = self.feature_engine.create_all_features(
                df_history,
                target_col=self.target_col,
                date_col='date'
            )

        # Get last date
        last_date = df_extended['date'].max()

        # Recursive forecasting
        forecasts = []

        for i in range(n_periods):
            # Create next month date
            next_date = last_date + pd.DateOffset(months=i+1)

            # Create row for next month
            next_row = pd.DataFrame([{
                'date': next_date,
                'year': next_date.year,
                'month': next_date.month,
                'quarter': next_date.quarter,
                'week': next_date.isocalendar().week,
                'day_of_year': next_date.dayofyear,
                'weekday': next_date.weekday()
            }])

            # Add to extended dataframe
            df_extended = pd.concat([df_extended, next_row], ignore_index=True)
            current_idx = len(df_extended) - 1

            # Update lag features using previous predictions
            lag_periods = self.config.get('features.lag_periods', [1, 3, 6, 12])
            for lag in lag_periods:
                lag_col = f'{self.target_col}_lag_{lag}'
                if current_idx >= lag:
                    df_extended.loc[current_idx, lag_col] = df_extended.loc[current_idx - lag, self.target_col]

            # Update rolling features
            rolling_windows = self.config.get('features.rolling_windows', [3, 6])
            for window in rolling_windows:
                mean_col = f'{self.target_col}_rolling_mean_{window}'
                std_col = f'{self.target_col}_rolling_std_{window}'

                if current_idx >= window - 1:
                    window_values = df_extended.loc[current_idx - window + 1:current_idx, self.target_col]
                    df_extended.loc[current_idx, mean_col] = window_values.mean()
                    df_extended.loc[current_idx, std_col] = window_values.std()

            # Update growth features
            growth_periods = [1, 3, 12]
            for period in growth_periods:
                growth_col = f'{self.target_col}_growth_{period}'
                if current_idx >= period:
                    current_val = df_extended.loc[current_idx - period, self.target_col]
                    if pd.notna(current_val) and current_val != 0:
                        prev_val = df_extended.loc[current_idx - period, self.target_col]
                        df_extended.loc[current_idx, growth_col] = (current_val - prev_val) / prev_val

            # Copy working_days if available (assume same as previous year same month)
            if 'working_days' in df_extended.columns:
                # Find same month from previous year
                same_month_prev_year = df_extended[
                    (df_extended['month'] == next_date.month) &
                    (df_extended['year'] == next_date.year - 1)
                ]
                if len(same_month_prev_year) > 0:
                    df_extended.loc[current_idx, 'working_days'] = same_month_prev_year.iloc[0]['working_days']

            # Make prediction
            X_pred = df_extended.loc[[current_idx], self.feature_cols]

            # Handle missing features
            if X_pred.isnull().any().any():
                logger.warning(f"  Month {i+1}: Missing features, filling with 0")
                X_pred = X_pred.fillna(0)

            prediction = self.model.predict(X_pred)[0]
            df_extended.loc[current_idx, self.target_col] = prediction

            forecasts.append({
                'date': next_date,
                self.target_col: prediction
            })

            logger.debug(f"  {next_date.strftime('%Y-%m')}: {prediction:,.0f}")

        # Create forecast dataframe
        df_forecast = pd.DataFrame(forecasts)

        logger.info(f"✅ Forecast complete")

        return df_forecast

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
        logger.info(f"Validating XGBoost on {len(df_val)} samples")

        # Create features for validation data
        df_val_features = self.feature_engine.create_all_features(
            df_val,
            target_col=target_col,
            date_col='date'
        )

        # Remove rows with NaN
        df_val_clean = df_val_features.dropna(subset=self.feature_cols + [target_col])

        if len(df_val_clean) == 0:
            logger.error("No valid validation samples after removing NaN")
            return {'mape': np.nan, 'mae': np.nan, 'rmse': np.nan, 'r2': np.nan}

        # Prepare validation data
        X_val = df_val_clean[self.feature_cols]
        y_val = df_val_clean[target_col]

        # Make predictions
        val_pred = self.model.predict(X_val)

        # Calculate metrics
        metrics = self._calculate_metrics(y_val.values, val_pred)

        self.validation_metrics = metrics

        logger.info(f"  Validation MAPE: {metrics['mape']:.2f}%")
        logger.info(f"  Validation MAE:  {metrics['mae']:,.2f}")

        return metrics
