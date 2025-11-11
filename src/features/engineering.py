"""Feature engineering for Traveco forecasting system"""

import pandas as pd
import numpy as np
from typing import List, Optional
from src.utils.config import ConfigLoader
from src.utils.logging_config import get_logger


logger = get_logger(__name__)


class TravecomFeatureEngine:
    """
    Feature engineering for time series forecasting

    Creates features from raw data:
    - Temporal features (year, month, quarter, etc.)
    - Lag features (1, 3, 6, 12 months)
    - Rolling statistics (mean, std)
    - Growth rates
    - Working days normalization
    """

    def __init__(self, config: Optional[ConfigLoader] = None):
        """
        Initialize feature engine

        Args:
            config: Configuration loader instance
        """
        self.config = config if config else ConfigLoader()
        self.lag_periods = self.config.get('features.lag_periods', [1, 3, 6, 12])
        self.rolling_windows = self.config.get('features.rolling_windows', [3, 6])

    def extract_temporal_features(
        self,
        df: pd.DataFrame,
        date_col: str = 'date'
    ) -> pd.DataFrame:
        """
        Extract temporal features from date column

        Features created:
        - year: Year (2022, 2023, etc.)
        - month: Month number (1-12)
        - quarter: Quarter (1-4)
        - week: ISO week number (1-53)
        - day_of_year: Day of year (1-365/366)
        - weekday: Day of week (0=Monday, 6=Sunday)

        Args:
            df: DataFrame with date column
            date_col: Name of date column

        Returns:
            DataFrame with temporal features added
        """
        logger.info(f"Extracting temporal features from '{date_col}'")

        df = df.copy()

        # Ensure datetime
        df[date_col] = pd.to_datetime(df[date_col])

        # Extract features
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['quarter'] = df[date_col].dt.quarter
        df['week'] = df[date_col].dt.isocalendar().week
        df['day_of_year'] = df[date_col].dt.dayofyear
        df['weekday'] = df[date_col].dt.weekday

        logger.info("  ✓ Created: year, month, quarter, week, day_of_year, weekday")

        return df

    def create_lag_features(
        self,
        df: pd.DataFrame,
        target_col: str,
        lag_periods: List[int] = None
    ) -> pd.DataFrame:
        """
        Create lag features for time series

        Args:
            df: DataFrame with time series data (must be sorted by date)
            target_col: Column to create lags for
            lag_periods: List of lag periods (default: from config)

        Returns:
            DataFrame with lag features added
        """
        if lag_periods is None:
            lag_periods = self.lag_periods

        logger.info(f"Creating lag features for '{target_col}': {lag_periods}")

        df = df.copy()

        for lag in lag_periods:
            feature_name = f'{target_col}_lag_{lag}'
            df[feature_name] = df[target_col].shift(lag)

        # Count non-null lags
        lag_cols = [f'{target_col}_lag_{lag}' for lag in lag_periods]
        non_null_count = df[lag_cols].notna().all(axis=1).sum()

        logger.info(f"  ✓ Created {len(lag_periods)} lag features ({non_null_count}/{len(df)} complete rows)")

        return df

    def create_rolling_features(
        self,
        df: pd.DataFrame,
        target_col: str,
        windows: List[int] = None
    ) -> pd.DataFrame:
        """
        Create rolling window statistics

        Features created for each window:
        - rolling_mean_{window}: Rolling mean
        - rolling_std_{window}: Rolling standard deviation

        Args:
            df: DataFrame with time series data (must be sorted by date)
            target_col: Column to create rolling features for
            windows: List of window sizes (default: from config)

        Returns:
            DataFrame with rolling features added
        """
        if windows is None:
            windows = self.rolling_windows

        logger.info(f"Creating rolling features for '{target_col}': {windows}")

        df = df.copy()

        for window in windows:
            # Rolling mean
            mean_name = f'{target_col}_rolling_mean_{window}'
            df[mean_name] = df[target_col].rolling(window=window, min_periods=1).mean()

            # Rolling std
            std_name = f'{target_col}_rolling_std_{window}'
            df[std_name] = df[target_col].rolling(window=window, min_periods=1).std()

        logger.info(f"  ✓ Created {len(windows) * 2} rolling features")

        return df

    def create_growth_features(
        self,
        df: pd.DataFrame,
        target_col: str,
        periods: List[int] = [1, 3, 12]
    ) -> pd.DataFrame:
        """
        Create growth rate features

        Args:
            df: DataFrame with time series data (must be sorted by date)
            target_col: Column to calculate growth for
            periods: List of periods for growth calculation

        Returns:
            DataFrame with growth features added
        """
        logger.info(f"Creating growth features for '{target_col}': {periods}")

        df = df.copy()

        for period in periods:
            feature_name = f'{target_col}_growth_{period}'
            df[feature_name] = df[target_col].pct_change(periods=period)

        logger.info(f"  ✓ Created {len(periods)} growth features")

        return df

    def create_all_features(
        self,
        df: pd.DataFrame,
        target_col: str,
        date_col: str = 'date',
        include_temporal: bool = True,
        include_lags: bool = True,
        include_rolling: bool = True,
        include_growth: bool = True
    ) -> pd.DataFrame:
        """
        Create all features for a target column

        This is the main method that orchestrates all feature engineering.

        Args:
            df: DataFrame with time series data
            target_col: Target column for forecasting
            date_col: Date column name
            include_temporal: Include temporal features
            include_lags: Include lag features
            include_rolling: Include rolling statistics
            include_growth: Include growth rates

        Returns:
            DataFrame with all features added
        """
        logger.info(f"Creating all features for '{target_col}'")

        df = df.copy()

        # Ensure sorted by date
        df = df.sort_values(date_col).reset_index(drop=True)

        # Temporal features
        if include_temporal:
            df = self.extract_temporal_features(df, date_col)

        # Lag features
        if include_lags:
            df = self.create_lag_features(df, target_col)

        # Rolling features
        if include_rolling:
            df = self.create_rolling_features(df, target_col)

        # Growth features
        if include_growth:
            df = self.create_growth_features(df, target_col)

        # Working days features (if present)
        if 'working_days' in df.columns:
            logger.info("  ✓ Including working_days as feature")

        logger.info(f"Feature engineering complete: {len(df.columns)} total columns")

        return df

    def get_feature_columns(
        self,
        df: pd.DataFrame,
        target_col: str,
        exclude_targets: bool = True
    ) -> List[str]:
        """
        Get list of feature columns (excluding targets and identifiers)

        Args:
            df: DataFrame
            target_col: Target column being forecasted
            exclude_targets: Exclude all target metrics

        Returns:
            List of feature column names
        """
        # Start with all columns
        feature_cols = df.columns.tolist()

        # Exclude identifiers and target
        exclude_cols = [
            target_col, 'date', 'year_month'
        ]

        # Exclude all target metrics if requested
        if exclude_targets:
            target_metrics = self.config.get('features.core_metrics', [])
            target_metrics += [
                'total_orders', 'total_km_billed', 'total_km_actual',
                'total_tours', 'total_drivers', 'internal_drivers',
                'revenue_total', 'external_drivers',
                'vehicle_km_cost', 'vehicle_time_cost', 'total_vehicle_cost',
                'personnel_costs', 'total_revenue_all'
            ]
            exclude_cols.extend(target_metrics)

            # Also exclude derived metrics
            derived_metrics = [
                'km_per_order', 'km_efficiency', 'revenue_per_order',
                'cost_per_order', 'profit_margin'
            ]
            exclude_cols.extend(derived_metrics)

        # Filter columns
        feature_cols = [col for col in feature_cols if col not in exclude_cols]

        logger.debug(f"Selected {len(feature_cols)} feature columns for '{target_col}'")

        return feature_cols

    def prepare_train_test_split(
        self,
        df: pd.DataFrame,
        target_col: str,
        train_end_date: str = None,
        val_start_date: str = None
    ) -> tuple:
        """
        Prepare train/test split for time series

        Args:
            df: DataFrame with features
            target_col: Target column
            train_end_date: End date for training (default: from config)
            val_start_date: Start date for validation (default: from config)

        Returns:
            Tuple of (X_train, X_val, y_train, y_val, train_dates, val_dates)
        """
        if train_end_date is None:
            train_end_date = self.config.get('training.train_end_date', '2024-06-30')

        if val_start_date is None:
            val_start_date = self.config.get('training.val_start_date', '2024-07-01')

        logger.info(f"Preparing train/test split: train until {train_end_date}, val from {val_start_date}")

        # Get feature columns
        feature_cols = self.get_feature_columns(df, target_col)

        # Remove rows with NaN in lag features (first N months)
        df_clean = df.dropna(subset=feature_cols + [target_col])

        logger.info(f"  Removed {len(df) - len(df_clean)} rows with missing features")

        # Split by date
        train_mask = df_clean['date'] <= pd.to_datetime(train_end_date)
        val_mask = df_clean['date'] >= pd.to_datetime(val_start_date)

        df_train = df_clean[train_mask]
        df_val = df_clean[val_mask]

        # Extract features and target
        X_train = df_train[feature_cols]
        X_val = df_val[feature_cols]
        y_train = df_train[target_col]
        y_val = df_val[target_col]

        train_dates = df_train['date']
        val_dates = df_val['date']

        logger.info(f"  Train: {len(X_train)} samples ({train_dates.min()} to {train_dates.max()})")
        logger.info(f"  Val:   {len(X_val)} samples ({val_dates.min()} to {val_dates.max()})")
        logger.info(f"  Features: {len(feature_cols)}")

        return X_train, X_val, y_train, y_val, train_dates, val_dates

    def convert_excel_dates(self, series: pd.Series) -> pd.Series:
        """
        Convert Excel date serial numbers to datetime

        Args:
            series: Series with date values (may be numeric or datetime)

        Returns:
            Series with datetime values
        """
        if pd.api.types.is_numeric_dtype(series):
            # Excel serial date: days since 1900-01-01
            return pd.to_datetime('1899-12-30') + pd.to_timedelta(series, 'D')
        else:
            return pd.to_datetime(series, errors='coerce')

    def identify_carrier_type(
        self,
        df: pd.DataFrame,
        carrier_col: str = 'Nummer.Spedition'
    ) -> pd.DataFrame:
        """
        Classify carriers as internal or external

        Args:
            df: DataFrame with carrier column
            carrier_col: Name of carrier number column

        Returns:
            DataFrame with carrier_type column added
        """
        logger.info(f"Classifying carrier types from '{carrier_col}'")

        df = df.copy()

        internal_max = self.config.get('filtering.internal_carrier_max', 8889)
        external_min = self.config.get('filtering.external_carrier_min', 9000)

        def classify(carrier_num):
            if pd.isna(carrier_num):
                return 'unknown'
            if carrier_num <= internal_max:
                return 'internal'
            elif carrier_num >= external_min:
                return 'external'
            return 'unknown'

        df['carrier_type'] = df[carrier_col].apply(classify)

        # Count by type
        type_counts = df['carrier_type'].value_counts()
        logger.info(f"  Internal: {type_counts.get('internal', 0):,}")
        logger.info(f"  External: {type_counts.get('external', 0):,}")
        logger.info(f"  Unknown:  {type_counts.get('unknown', 0):,}")

        return df
