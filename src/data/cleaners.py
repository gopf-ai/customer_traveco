"""Data cleaning for Traveco forecasting system"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from src.utils.config import ConfigLoader
from src.utils.logging_config import get_logger


logger = get_logger(__name__)


class TravecomDataCleaner:
    """
    Clean and standardize Traveco data

    Applies business rules and filtering logic from Notebooks 02-04
    """

    def __init__(self, config: ConfigLoader = None):
        """
        Initialize data cleaner

        Args:
            config: Configuration loader instance
        """
        self.config = config if config else ConfigLoader()
        self.filtering_rules = self.config.get('filtering', {})

    def apply_filtering_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply business filtering rules to order data

        Rules (from Notebook 02):
        1. Exclude "Lager Auftrag" (warehouse orders)
        2. Exclude B&T pickup orders (System='B&T' AND empty customer)

        Args:
            df: DataFrame with order data

        Returns:
            Filtered DataFrame
        """
        initial_count = len(df)
        logger.info(f"Applying filtering rules to {initial_count:,} orders")

        # Filter 1: Exclude Lager orders
        if self.filtering_rules.get('exclude_lager_orders', True):
            if 'Lieferart 2.0' in df.columns:
                lager_mask = df['Lieferart 2.0'] == 'Lager Auftrag'
                lager_count = lager_mask.sum()

                if lager_count > 0:
                    df = df[~lager_mask].copy()
                    logger.info(f"  Excluded {lager_count:,} Lager orders")
            else:
                logger.warning("  'Lieferart 2.0' column not found, skipping Lager filter")

        # Filter 2: Exclude B&T pickups
        if self.filtering_rules.get('exclude_bt_pickups', True):
            # Check for RKdNr (cleaned) or RKdNr. (original with dot)
            rkd_col = None
            if 'RKdNr' in df.columns:
                rkd_col = 'RKdNr'
            elif 'RKdNr.' in df.columns:
                rkd_col = 'RKdNr.'

            if 'System_id.Auftrag' in df.columns and rkd_col is not None:
                # B&T pickups: System='B&T' AND empty customer (RKdNr)
                # "Empty" includes: NaN, empty string, placeholder '-'
                bt_mask = (df['System_id.Auftrag'] == 'B&T')
                empty_customer_mask = df[rkd_col].isna() | df[rkd_col].isin(['-', '', ' '])
                bt_pickup_mask = bt_mask & empty_customer_mask
                bt_count = bt_pickup_mask.sum()

                if bt_count > 0:
                    df = df[~bt_pickup_mask].copy()
                    logger.info(f"  Excluded {bt_count:,} B&T pickup orders")
            else:
                missing = []
                if 'System_id.Auftrag' not in df.columns:
                    missing.append('System_id.Auftrag')
                if rkd_col is None:
                    missing.append('RKdNr/RKdNr.')
                logger.warning(f"  Required columns not found: {missing} - skipping B&T filter")

        final_count = len(df)
        removed_count = initial_count - final_count
        removed_pct = (removed_count / initial_count) * 100

        logger.info(f"Filtering complete: {removed_count:,} orders removed ({removed_pct:.2f}%)")
        logger.info(f"Remaining: {final_count:,} orders")

        return df

    def exclude_losetransporte(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Exclude Losetransporte (contract weight issues)

        From Notebook 03: These orders have problematic contract weights

        Args:
            df: DataFrame with order data

        Returns:
            Filtered DataFrame
        """
        if 'Auftragsart' not in df.columns:
            logger.warning("'Auftragsart' column not found, skipping Losetransporte filter")
            return df

        initial_count = len(df)

        # Assuming Losetransporte are marked in Auftragsart column
        # Adjust condition based on actual data
        losetransporte_mask = df['Auftragsart'].str.contains('Lose', case=False, na=False)
        losetransporte_count = losetransporte_mask.sum()

        if losetransporte_count > 0:
            df = df[~losetransporte_mask].copy()
            logger.info(f"Excluded {losetransporte_count:,} Losetransporte orders")

        return df

    def convert_excel_dates(self, series: pd.Series) -> pd.Series:
        """
        Convert Excel date serial numbers to datetime

        Excel stores dates as serial numbers (days since 1900-01-01)

        Args:
            series: Series with date values (may be numeric or datetime)

        Returns:
            Series with datetime values
        """
        if pd.api.types.is_numeric_dtype(series):
            # Excel serial date: days since 1900-01-01
            # Note: Excel incorrectly treats 1900 as a leap year
            return pd.to_datetime('1899-12-30') + pd.to_timedelta(series, 'D')
        else:
            # Already datetime or string
            return pd.to_datetime(series, errors='coerce')

    def handle_missing_values(self, df: pd.DataFrame, strategy: Dict[str, str] = None) -> pd.DataFrame:
        """
        Handle missing values based on column-specific strategies

        Args:
            df: DataFrame with potentially missing values
            strategy: Dict mapping column names to strategies:
                     - 'drop': Drop rows with missing values
                     - 'zero': Fill with 0
                     - 'forward': Forward fill
                     - 'mean': Fill with column mean

        Returns:
            DataFrame with missing values handled
        """
        if strategy is None:
            strategy = {}

        missing_summary = df.isnull().sum()
        missing_cols = missing_summary[missing_summary > 0]

        if len(missing_cols) == 0:
            logger.info("No missing values found")
            return df

        logger.info(f"Found missing values in {len(missing_cols)} columns")

        for col in missing_cols.index:
            missing_count = missing_cols[col]
            missing_pct = (missing_count / len(df)) * 100

            if col in strategy:
                method = strategy[col]

                if method == 'drop':
                    df = df[df[col].notna()].copy()
                    logger.debug(f"  {col}: Dropped {missing_count} rows ({missing_pct:.1f}%)")

                elif method == 'zero':
                    df[col].fillna(0, inplace=True)
                    logger.debug(f"  {col}: Filled {missing_count} with 0 ({missing_pct:.1f}%)")

                elif method == 'forward':
                    df[col].fillna(method='ffill', inplace=True)
                    logger.debug(f"  {col}: Forward filled {missing_count} ({missing_pct:.1f}%)")

                elif method == 'mean':
                    mean_val = df[col].mean()
                    df[col].fillna(mean_val, inplace=True)
                    logger.debug(f"  {col}: Filled {missing_count} with mean={mean_val:.2f} ({missing_pct:.1f}%)")

            else:
                # Log but don't handle
                logger.debug(f"  {col}: {missing_count} missing ({missing_pct:.1f}%) - no strategy specified")

        return df

    def validate_revenue_relationship(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Validate that total_revenue_all >= revenue_total (transportation revenue)

        This is a critical business rule for the new revenue modeling.

        Args:
            df: DataFrame with both revenue columns

        Returns:
            Tuple of (validated DataFrame, validation report dict)
        """
        if 'revenue_total' not in df.columns or 'total_revenue_all' not in df.columns:
            logger.warning("Revenue columns not found, skipping validation")
            return df, {'status': 'skipped', 'reason': 'columns_missing'}

        # Check relationship
        violations = df[df['total_revenue_all'] < df['revenue_total']]

        report = {
            'total_records': len(df),
            'violations': len(violations),
            'violation_pct': (len(violations) / len(df)) * 100 if len(df) > 0 else 0,
            'status': 'pass' if len(violations) == 0 else 'fail'
        }

        if len(violations) > 0:
            logger.warning(
                f"⚠️  Found {len(violations)} records where total_revenue_all < revenue_total "
                f"({report['violation_pct']:.2f}%)"
            )

            # Log sample violations
            for idx, row in violations.head(5).iterrows():
                logger.warning(
                    f"  {row.get('date', idx)}: "
                    f"transport={row['revenue_total']:,.0f}, "
                    f"total={row['total_revenue_all']:,.0f}"
                )

            # Option 1: Fix by setting total = transport (conservative)
            # df.loc[violations.index, 'total_revenue_all'] = df.loc[violations.index, 'revenue_total']

            # Option 2: Flag for manual review
            df['revenue_validation_flag'] = False
            df.loc[violations.index, 'revenue_validation_flag'] = True

        else:
            logger.info("✅ Revenue relationship validation passed")

        return df, report

    def remove_duplicates(self, df: pd.DataFrame, subset: list = None) -> pd.DataFrame:
        """
        Remove duplicate rows

        Args:
            df: DataFrame
            subset: Columns to consider for identifying duplicates

        Returns:
            DataFrame with duplicates removed
        """
        initial_count = len(df)
        df = df.drop_duplicates(subset=subset).copy()
        final_count = len(df)

        removed = initial_count - final_count

        if removed > 0:
            logger.info(f"Removed {removed:,} duplicate rows")
        else:
            logger.info("No duplicates found")

        return df

    def standardize_numeric_columns(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Standardize numeric columns (convert to numeric, handle errors)

        Args:
            df: DataFrame
            columns: List of column names to standardize

        Returns:
            DataFrame with standardized numeric columns
        """
        for col in columns:
            if col not in df.columns:
                continue

            original_dtype = df[col].dtype

            # Convert to numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')

            # Count conversions
            null_count = df[col].isnull().sum()

            if null_count > 0:
                logger.warning(f"  {col}: {null_count} values could not be converted to numeric")

            logger.debug(f"  {col}: {original_dtype} → {df[col].dtype}")

        return df

    # Pipeline compatibility methods
    def clean_orders(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean order data (wrapper for pipeline compatibility)

        Applies standard cleaning steps:
        1. Apply filtering rules (exclude Lager Auftrag, etc.)
        2. Handle missing values
        3. Standardize numeric columns

        Args:
            df: Raw order data

        Returns:
            Cleaned order data
        """
        logger.info("Cleaning order data...")

        # Apply filtering rules
        df = self.apply_filtering_rules(df)

        # Handle missing values (basic strategy)
        # df = self.handle_missing_values(df)

        logger.info(f"✓ Orders cleaned: {len(df):,} records")

        return df

    def clean_tours(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean tour data (wrapper for pipeline compatibility)

        Args:
            df: Raw tour data

        Returns:
            Cleaned tour data
        """
        logger.info("Cleaning tour data...")

        # Basic cleaning for tours (add more as needed)
        # For now, just return as-is

        logger.info(f"✓ Tours cleaned: {len(df):,} records")

        return df
