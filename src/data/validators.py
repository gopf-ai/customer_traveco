"""Data validation for Traveco forecasting system"""

import pandas as pd
import numpy as np
from typing import Dict, List
from src.utils.logging_config import get_logger


logger = get_logger(__name__)


class DataValidator:
    """
    Validate data quality and business rules

    Performs comprehensive checks on:
    - Data completeness
    - Value ranges
    - Business rule compliance
    - Outlier detection
    """

    def __init__(self, config: dict = None):
        """
        Initialize validator

        Args:
            config: Validation configuration (thresholds, rules)
        """
        self.config = config if config else {}
        self.validation_results = {}

    def validate_completeness(self, df: pd.DataFrame, required_columns: List[str]) -> Dict:
        """
        Check if required columns exist and have sufficient data

        Args:
            df: DataFrame to validate
            required_columns: List of required column names

        Returns:
            Validation report dict
        """
        logger.info("Validating data completeness...")

        missing_columns = [col for col in required_columns if col not in df.columns]
        present_columns = [col for col in required_columns if col in df.columns]

        # Check completeness for present columns
        completeness = {}
        for col in present_columns:
            non_null_count = df[col].notna().sum()
            completeness[col] = {
                'records': len(df),
                'non_null': non_null_count,
                'null': len(df) - non_null_count,
                'completeness_pct': (non_null_count / len(df)) * 100 if len(df) > 0 else 0
            }

        report = {
            'status': 'pass' if len(missing_columns) == 0 else 'fail',
            'total_columns': len(required_columns),
            'missing_columns': missing_columns,
            'present_columns': len(present_columns),
            'completeness': completeness
        }

        if missing_columns:
            logger.error(f"❌ Missing required columns: {missing_columns}")
        else:
            logger.info("✅ All required columns present")

        # Log low completeness
        for col, stats in completeness.items():
            if stats['completeness_pct'] < 95:
                logger.warning(
                    f"⚠️  {col}: {stats['completeness_pct']:.1f}% complete "
                    f"({stats['null']} missing values)"
                )

        self.validation_results['completeness'] = report
        return report

    def validate_date_range(self, df: pd.DataFrame, date_col: str,
                           expected_start: str = None, expected_end: str = None) -> Dict:
        """
        Validate date range coverage

        Args:
            df: DataFrame with date column
            date_col: Name of date column
            expected_start: Expected start date (YYYY-MM-DD)
            expected_end: Expected end date (YYYY-MM-DD)

        Returns:
            Validation report dict
        """
        logger.info(f"Validating date range for '{date_col}'...")

        if date_col not in df.columns:
            return {'status': 'error', 'message': f"Date column '{date_col}' not found"}

        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

        actual_start = df[date_col].min()
        actual_end = df[date_col].max()

        report = {
            'status': 'pass',
            'actual_start': str(actual_start),
            'actual_end': str(actual_end),
            'date_count': df[date_col].notna().sum(),
            'missing_dates': df[date_col].isna().sum()
        }

        if expected_start:
            expected_start_dt = pd.to_datetime(expected_start)
            if actual_start > expected_start_dt:
                report['status'] = 'warning'
                logger.warning(f"⚠️  Start date {actual_start} is after expected {expected_start}")

        if expected_end:
            expected_end_dt = pd.to_datetime(expected_end)
            if actual_end < expected_end_dt:
                report['status'] = 'warning'
                logger.warning(f"⚠️  End date {actual_end} is before expected {expected_end}")

        logger.info(f"  Date range: {actual_start} to {actual_end}")

        self.validation_results['date_range'] = report
        return report

    def detect_outliers(self, df: pd.DataFrame, column: str,
                       std_threshold: float = 3.0) -> Dict:
        """
        Detect outliers using standard deviation method

        Args:
            df: DataFrame
            column: Column to check for outliers
            std_threshold: Number of standard deviations for outlier threshold

        Returns:
            Validation report dict with outlier information
        """
        if column not in df.columns:
            return {'status': 'error', 'message': f"Column '{column}' not found"}

        values = df[column].dropna()

        if len(values) == 0:
            return {'status': 'error', 'message': 'No non-null values to analyze'}

        mean = values.mean()
        std = values.std()

        lower_bound = mean - (std_threshold * std)
        upper_bound = mean + (std_threshold * std)

        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

        report = {
            'status': 'pass' if len(outliers) == 0 else 'warning',
            'column': column,
            'mean': float(mean),
            'std': float(std),
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound),
            'outlier_count': len(outliers),
            'outlier_pct': (len(outliers) / len(df)) * 100 if len(df) > 0 else 0
        }

        if len(outliers) > 0:
            logger.warning(
                f"⚠️  {column}: Found {len(outliers)} outliers ({report['outlier_pct']:.2f}%) "
                f"outside {std_threshold} std deviations"
            )

            # Log sample outliers
            outlier_samples = outliers.nlargest(3, column)[column].tolist()
            logger.warning(f"  Top outliers: {[f'{v:,.0f}' for v in outlier_samples]}")

        self.validation_results[f'outliers_{column}'] = report
        return report

    def validate_monthly_completeness(self, df: pd.DataFrame, date_col: str,
                                     expected_months: int = None) -> Dict:
        """
        Check for missing months in time series

        Args:
            df: DataFrame with date column
            date_col: Name of date column
            expected_months: Expected number of months

        Returns:
            Validation report dict
        """
        logger.info("Validating monthly completeness...")

        if date_col not in df.columns:
            return {'status': 'error', 'message': f"Date column '{date_col}' not found"}

        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

        # Extract year-month
        df['year_month'] = df[date_col].dt.to_period('M')

        # Count unique months
        actual_months = df['year_month'].nunique()

        # Generate expected months range
        start_date = df[date_col].min()
        end_date = df[date_col].max()

        expected_range = pd.period_range(start_date, end_date, freq='M')
        expected_months_count = len(expected_range)

        # Find missing months
        present_months = set(df['year_month'].unique())
        all_months = set(expected_range)
        missing_months = all_months - present_months

        report = {
            'status': 'pass' if len(missing_months) == 0 else 'warning',
            'actual_months': actual_months,
            'expected_months': expected_months_count,
            'missing_months': sorted([str(m) for m in missing_months]),
            'missing_count': len(missing_months),
            'completeness_pct': (actual_months / expected_months_count) * 100 if expected_months_count > 0 else 0
        }

        if missing_months:
            logger.warning(f"⚠️  Missing {len(missing_months)} months: {report['missing_months']}")
        else:
            logger.info("✅ All months present in date range")

        self.validation_results['monthly_completeness'] = report
        return report

    def validate_value_ranges(self, df: pd.DataFrame, rules: Dict[str, Dict]) -> Dict:
        """
        Validate that values fall within expected ranges

        Args:
            df: DataFrame
            rules: Dict mapping column names to range rules
                  e.g., {'revenue_total': {'min': 0, 'max': 1000000}}

        Returns:
            Validation report dict
        """
        logger.info("Validating value ranges...")

        violations = {}

        for column, range_rule in rules.items():
            if column not in df.columns:
                logger.warning(f"Column '{column}' not found, skipping range validation")
                continue

            min_val = range_rule.get('min')
            max_val = range_rule.get('max')

            col_violations = []

            if min_val is not None:
                below_min = df[df[column] < min_val]
                if len(below_min) > 0:
                    col_violations.append({
                        'type': 'below_min',
                        'count': len(below_min),
                        'min_expected': min_val,
                        'min_actual': float(df[column].min())
                    })

            if max_val is not None:
                above_max = df[df[column] > max_val]
                if len(above_max) > 0:
                    col_violations.append({
                        'type': 'above_max',
                        'count': len(above_max),
                        'max_expected': max_val,
                        'max_actual': float(df[column].max())
                    })

            if col_violations:
                violations[column] = col_violations
                for v in col_violations:
                    logger.warning(
                        f"⚠️  {column}: {v['count']} values {v['type']} "
                        f"(expected: {v.get('min_expected') or v.get('max_expected')})"
                    )

        report = {
            'status': 'pass' if len(violations) == 0 else 'warning',
            'rules_checked': len(rules),
            'violations': violations
        }

        self.validation_results['value_ranges'] = report
        return report

    def generate_validation_report(self) -> pd.DataFrame:
        """
        Generate comprehensive validation report

        Returns:
            DataFrame with all validation results
        """
        if not self.validation_results:
            logger.warning("No validation results to report")
            return pd.DataFrame()

        report_rows = []

        for check_name, result in self.validation_results.items():
            report_rows.append({
                'check': check_name,
                'status': result.get('status', 'unknown'),
                'details': str(result)
            })

        df_report = pd.DataFrame(report_rows)

        logger.info(f"\n{'='*50}")
        logger.info("VALIDATION SUMMARY")
        logger.info(f"{'='*50}")

        for status in ['fail', 'warning', 'pass']:
            count = (df_report['status'] == status).sum()
            if count > 0:
                logger.info(f"{status.upper()}: {count} checks")

        return df_report

    def validate_dataframe(self, df: pd.DataFrame, dataset_name: str = 'data') -> Dict:
        """
        Run basic validation on a DataFrame (pipeline compatibility wrapper)

        Args:
            df: DataFrame to validate
            dataset_name: Name of the dataset for logging

        Returns:
            Dictionary with validation summary
        """
        logger.info(f"Validating {dataset_name}...")

        # Basic validation metrics
        total_records = len(df)
        complete_records = df.dropna().shape[0]
        missing_values = df.isnull().sum().sum()

        # Check for date column
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        if date_cols:
            date_col = date_cols[0]
            # Run date validation
            try:
                self.validate_date_range(df, date_col)
            except:
                pass

        validation_summary = {
            'total_records': int(total_records),
            'complete_records': int(complete_records),
            'missing_values': int(missing_values),
            'outliers_detected': 0  # Placeholder
        }

        logger.info(f"✓ Validation complete for {dataset_name}")

        return validation_summary
