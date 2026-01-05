"""Financial data preparation pipeline

Loads financial Excel/CSV files and prepares monthly time series for forecasting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
import os
import re
from src.utils.logging_config import get_logger
from src.utils.config import ConfigLoader


logger = get_logger(__name__)


class FinancialDataPrep:
    """
    Prepare financial data for forecasting

    Steps:
    1. Load financial files from all years (wide format)
    2. Extract target metrics by Sachkonto
    3. Transform to long format (monthly time series)
    4. Add working days feature
    5. Save to intermediate directory
    """

    # Target metrics with their Sachkonto mappings
    METRIC_MAPPING = {
        'total_betriebsertrag': ['0140', 'Total Betriebsertrag'],
        'total_revenue': ['35060000', '35070000'],  # Transport + Logistics
        'personnel_costs': ['0151', 'Personalaufwand'],
        'external_driver_costs': ['6280', 'Ausgangsfrachten LKW'],
        'ebt': ['0110', 'Ergebnis vor Steuern']
    }

    def __init__(self, config: Optional[ConfigLoader] = None):
        """Initialize data preparation"""
        self.config = config if config else ConfigLoader()
        self.raw_path = Path(self.config.get('data.raw_path', 'data/raw'))
        self.intermediate_path = Path(self.config.get('data.intermediate_path', 'data/intermediate'))

        if not self.raw_path.is_absolute():
            project_root = Path(__file__).parent.parent.parent
            self.raw_path = project_root / self.raw_path

        if not self.intermediate_path.is_absolute():
            project_root = Path(__file__).parent.parent.parent
            self.intermediate_path = project_root / self.intermediate_path

        self.intermediate_path.mkdir(parents=True, exist_ok=True)

    def run(self) -> pd.DataFrame:
        """
        Run the complete financial data preparation pipeline

        Returns:
            Monthly time series DataFrame
        """
        logger.info("=" * 60)
        logger.info("FINANCIAL DATA PREPARATION")
        logger.info("=" * 60)

        # Step 1: Load all financial data
        all_data = self._load_all_years()

        # Step 2: Transform to long format
        df_monthly = self._transform_to_long(all_data)

        # Step 3: Add working days
        df_monthly = self._add_working_days(df_monthly)

        # Step 4: Save to intermediate
        output_path = self.intermediate_path / 'financial_time_series.csv'
        df_monthly.to_csv(output_path, index=False)
        logger.info(f"Saved financial time series to: {output_path}")

        return df_monthly

    def _load_all_years(self) -> Dict[int, pd.DataFrame]:
        """Load financial data from all years"""
        years = self.config.get('data.years', [2022, 2023, 2024, 2025])
        logger.info(f"Loading financial data from years: {years}")

        all_data = {}

        for year in years:
            # Try multiple paths
            paths_to_try = [
                self.raw_path / str(year) / f'{year} Finanzen' / f'{year}.csv',
                self.raw_path / str(year) / f'{year} Finanzen' / f'{year}.xlsx',
                self.raw_path / str(year) / 'Finanzen' / f'{year}.csv',
                self.raw_path / str(year) / 'Finanzen' / f'{year}.xlsx'
            ]

            for file_path in paths_to_try:
                if file_path.exists():
                    logger.info(f"  Loading: {file_path}")
                    try:
                        df = self._load_financial_file(file_path)
                        all_data[year] = df
                        break
                    except Exception as e:
                        logger.error(f"Error loading {file_path}: {e}")

            if year not in all_data:
                logger.warning(f"No financial data found for {year}")

        return all_data

    def _load_financial_file(self, file_path: Path) -> pd.DataFrame:
        """Load a single financial file"""
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path, encoding='utf-8-sig')
        else:
            df = pd.read_excel(file_path)

        # Clean column names
        df.columns = [str(col).strip() for col in df.columns]

        return df

    def _transform_to_long(self, all_data: Dict[int, pd.DataFrame]) -> pd.DataFrame:
        """Transform wide financial data to long format time series"""
        logger.info("Transforming to long format...")

        # Month name mapping (German)
        month_mapping = {
            'Januar': 1, 'Februar': 2, 'März': 3, 'April': 4,
            'Mai': 5, 'Juni': 6, 'Juli': 7, 'August': 8,
            'September': 9, 'Oktober': 10, 'November': 11, 'Dezember': 12
        }

        records = []

        for year, df in all_data.items():
            logger.info(f"  Processing {year}...")

            # Find Sachkonto column
            sachkonto_col = self._find_column(df, ['Sachkonto', 'Account', ''])

            for metric_name, identifiers in self.METRIC_MAPPING.items():
                # Find matching rows
                metric_rows = self._find_metric_rows(df, sachkonto_col, identifiers)

                if metric_rows.empty:
                    logger.warning(f"    Metric {metric_name} not found for {year}")
                    continue

                # Extract monthly values
                for month_name, month_num in month_mapping.items():
                    if month_name in df.columns:
                        # Sum across all matching rows (for revenue that has multiple components)
                        month_values = metric_rows[month_name]

                        # Convert Swiss number format
                        total_value = 0
                        for val in month_values:
                            total_value += self._parse_swiss_number(val)

                        # Financial values are negative for revenue/income in P&L
                        # Normalize: make revenue positive
                        if metric_name in ['total_betriebsertrag', 'total_revenue']:
                            total_value = abs(total_value)

                        records.append({
                            'date': pd.Timestamp(year=year, month=month_num, day=1),
                            'year': year,
                            'month': month_num,
                            'metric': metric_name,
                            'value': total_value
                        })

        # Convert to wide format (one column per metric)
        df_long = pd.DataFrame(records)

        if df_long.empty:
            logger.warning("No financial data extracted")
            return pd.DataFrame()

        # Pivot to wide format
        df_wide = df_long.pivot_table(
            index=['date', 'year', 'month'],
            columns='metric',
            values='value',
            aggfunc='sum'
        ).reset_index()

        # Flatten column names
        df_wide.columns = [col if col != '' else col for col in df_wide.columns]

        df_wide = df_wide.sort_values('date').reset_index(drop=True)

        logger.info(f"Created time series: {len(df_wide)} months")
        logger.info(f"  Metrics: {[col for col in df_wide.columns if col not in ['date', 'year', 'month']]}")

        return df_wide

    def _find_metric_rows(
        self,
        df: pd.DataFrame,
        sachkonto_col: Optional[str],
        identifiers: List[str]
    ) -> pd.DataFrame:
        """Find rows matching metric identifiers"""
        if df.empty:
            return pd.DataFrame()

        mask = pd.Series([False] * len(df))

        for identifier in identifiers:
            # Try Sachkonto column
            if sachkonto_col and sachkonto_col in df.columns:
                col_match = df[sachkonto_col].astype(str).str.contains(identifier, case=False, na=False)
                mask = mask | col_match

            # Try first column (often unnamed with Sachkonto)
            first_col = df.columns[0]
            first_match = df[first_col].astype(str).str.contains(identifier, case=False, na=False)
            mask = mask | first_match

            # Try second column (Geschäftsperiode/description)
            if len(df.columns) > 1:
                second_col = df.columns[1]
                second_match = df[second_col].astype(str).str.contains(identifier, case=False, na=False)
                mask = mask | second_match

        return df[mask]

    def _find_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Find column from candidates"""
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def _parse_swiss_number(self, value) -> float:
        """Parse Swiss number format (1'234,56 or -1,234.56)"""
        if pd.isna(value):
            return 0.0

        if isinstance(value, (int, float)):
            return float(value)

        # Convert to string and clean
        s = str(value).strip()

        if not s or s == '-':
            return 0.0

        # Remove thousand separators (', or .)
        s = s.replace("'", "").replace(" ", "")

        # Handle Swiss format: comma as decimal separator
        # Check if comma is used as decimal (has digits after it)
        if ',' in s:
            parts = s.split(',')
            if len(parts) == 2 and len(parts[1]) <= 2:
                # Comma is decimal separator
                s = s.replace(',', '.')
            else:
                # Comma is thousand separator
                s = s.replace(',', '')

        try:
            return float(s)
        except ValueError:
            return 0.0

    def _add_working_days(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add working days from external file"""
        logger.info("Adding working days...")

        working_days_file = self.raw_path / 'TRAVECO_Arbeitstage_2022-laufend_für gopf.com_hb v2.xlsx'

        if not working_days_file.exists():
            working_days_file = self.raw_path / 'TRAVECO_Arbeitstage_2022-laufend_für gopf.com_hb v1.xlsx'

        if not working_days_file.exists():
            logger.warning("Working days file not found - skipping")
            return df

        try:
            df_wd = pd.read_excel(working_days_file)

            # Transform from wide to long format
            month_mapping = {
                'Januar': 1, 'Februar': 2, 'März': 3, 'April': 4,
                'Mai': 5, 'Juni': 6, 'Juli': 7, 'August': 8,
                'September': 9, 'Oktober': 10, 'November': 11, 'Dezember': 12
            }

            wd_records = []
            for _, row in df_wd.iterrows():
                year = row['Jahr']
                for month_name, month_num in month_mapping.items():
                    if month_name in row and pd.notna(row[month_name]):
                        wd_records.append({
                            'year': int(year),
                            'month': month_num,
                            'working_days': int(row[month_name])
                        })

            df_working_days = pd.DataFrame(wd_records)

            # Merge
            df = df.merge(df_working_days, on=['year', 'month'], how='left')
            logger.info(f"  Added working days for {df['working_days'].notna().sum()} months")

        except Exception as e:
            logger.error(f"Error loading working days: {e}")

        return df


def run():
    """Entry point for pipeline"""
    prep = FinancialDataPrep()
    return prep.run()


if __name__ == '__main__':
    run()
