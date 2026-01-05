"""Operational data preparation pipeline

Loads raw Excel files and prepares monthly time series for forecasting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List
import os
from src.utils.logging_config import get_logger
from src.utils.config import ConfigLoader


logger = get_logger(__name__)


class OperationalDataPrep:
    """
    Prepare operational data for forecasting

    Steps:
    1. Load raw order files from all years
    2. Apply filtering rules
    3. Aggregate to monthly time series
    4. Add working days feature
    5. Save to intermediate directory
    """

    # Target metrics for operational forecasting
    METRICS = [
        'total_orders',
        'total_km_billed',
        'total_km_actual',
        'total_tours',
        'total_drivers',
        'revenue_total',
        'external_drivers',
        'vehicle_km_cost',
        'vehicle_time_cost',
        'total_vehicle_cost'
    ]

    def __init__(self, config: Optional[ConfigLoader] = None):
        """Initialize data preparation"""
        self.config = config if config else ConfigLoader()
        self.raw_path = Path(self.config.get('data.raw_path', 'data/raw'))
        self.intermediate_path = Path(self.config.get('data.intermediate_path', 'data/intermediate'))

        # Ensure paths exist
        if not self.raw_path.is_absolute():
            project_root = Path(__file__).parent.parent.parent
            self.raw_path = project_root / self.raw_path

        if not self.intermediate_path.is_absolute():
            project_root = Path(__file__).parent.parent.parent
            self.intermediate_path = project_root / self.intermediate_path

        self.intermediate_path.mkdir(parents=True, exist_ok=True)

    def run(self) -> pd.DataFrame:
        """
        Run the complete data preparation pipeline

        Returns:
            Monthly time series DataFrame
        """
        logger.info("=" * 60)
        logger.info("OPERATIONAL DATA PREPARATION")
        logger.info("=" * 60)

        # Step 1: Load all raw data
        df_orders = self._load_all_orders()

        # Step 2: Load tour data for vehicle costs
        df_tours = self._load_all_tours()

        # Step 3: Apply filtering
        df_filtered = self._apply_filters(df_orders)

        # Step 4: Aggregate to monthly
        df_monthly = self._aggregate_monthly(df_filtered, df_tours)

        # Step 5: Add working days
        df_monthly = self._add_working_days(df_monthly)

        # Step 6: Save to intermediate
        output_path = self.intermediate_path / 'operational_time_series.csv'
        df_monthly.to_csv(output_path, index=False)
        logger.info(f"Saved operational time series to: {output_path}")

        return df_monthly

    def _load_all_orders(self) -> pd.DataFrame:
        """Load order data from all years"""
        years = self.config.get('data.years', [2022, 2023, 2024, 2025])
        logger.info(f"Loading orders from years: {years}")

        all_dfs = []
        for year in years:
            year_path = self.raw_path / str(year)
            if not year_path.exists():
                logger.warning(f"Year directory not found: {year_path}")
                continue

            # Find order analysis files
            order_files = sorted([f for f in os.listdir(year_path)
                                 if 'QS Auftragsanalyse' in f and f.endswith(('.xlsx', '.xlsb'))])

            for file_name in order_files:
                file_path = year_path / file_name
                logger.info(f"  Loading: {year}/{file_name}")

                try:
                    if file_name.endswith('.xlsb'):
                        df = pd.read_excel(file_path, engine='pyxlsb')
                    else:
                        df = pd.read_excel(file_path)

                    # Clean column names (remove trailing dots)
                    df.columns = [col.rstrip('.').strip() for col in df.columns]

                    df['_source_year'] = year
                    df['_source_file'] = file_name
                    all_dfs.append(df)

                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")

        if not all_dfs:
            raise ValueError("No order files found")

        df_combined = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"Loaded {len(df_combined):,} orders total")

        return df_combined

    def _load_all_tours(self) -> pd.DataFrame:
        """Load tour data from all years"""
        years = self.config.get('data.years', [2022, 2023, 2024, 2025])
        logger.info(f"Loading tours from years: {years}")

        all_dfs = []
        for year in years:
            year_path = self.raw_path / str(year)
            if not year_path.exists():
                continue

            # Find tour file
            tour_files = [f for f in os.listdir(year_path)
                         if 'QS Tourenaufstellung' in f and f.endswith('.xlsx')]

            if not tour_files:
                logger.warning(f"No tour file found for {year}")
                continue

            file_path = year_path / tour_files[0]
            logger.info(f"  Loading: {year}/{tour_files[0]}")

            try:
                df = pd.read_excel(file_path)
                df.columns = [col.rstrip('.').strip() for col in df.columns]
                df['_source_year'] = year
                all_dfs.append(df)

            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")

        if not all_dfs:
            logger.warning("No tour files found - vehicle costs will be unavailable")
            return pd.DataFrame()

        df_combined = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"Loaded {len(df_combined):,} tours total")

        return df_combined

    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply data filtering rules"""
        logger.info("Applying filtering rules...")

        initial_count = len(df)

        # Convert date column
        date_col = self._find_date_column(df)
        if date_col:
            df['date'] = self._convert_dates(df[date_col])
        else:
            logger.warning("No date column found - using source year")
            df['date'] = pd.to_datetime(df['_source_year'].astype(str) + '-01-01')

        # Filter: Exclude "Lager Auftrag" (warehouse orders)
        auftrag_col = self._find_column(df, ['Auftrag', 'Auftragsart'])
        if auftrag_col:
            df = df[~df[auftrag_col].astype(str).str.contains('Lager', case=False, na=False)]
            logger.info(f"  After excluding Lager orders: {len(df):,}")

        # Filter: Exclude missing carrier numbers
        carrier_col = self._find_column(df, ['Nummer.Spedition', 'Spedition'])
        if carrier_col:
            df = df[df[carrier_col].notna()]
            logger.info(f"  After excluding missing carriers: {len(df):,}")

        filtered_count = len(df)
        logger.info(f"Filtered: {initial_count:,} -> {filtered_count:,} ({filtered_count/initial_count*100:.1f}% retained)")

        return df

    def _aggregate_monthly(self, df_orders: pd.DataFrame, df_tours: pd.DataFrame) -> pd.DataFrame:
        """Aggregate to monthly time series"""
        logger.info("Aggregating to monthly time series...")

        # Ensure date column
        df_orders['year'] = df_orders['date'].dt.year
        df_orders['month'] = df_orders['date'].dt.month

        # Find relevant columns
        km_billed_col = self._find_column(df_orders, ['Distanz_BE.Auftrag', 'Distanz', 'KM'])
        revenue_col = self._find_column(df_orders, ['Umsatz.Auftrag', 'Umsatz', 'Revenue'])
        carrier_col = self._find_column(df_orders, ['Nummer.Spedition', 'Spedition'])

        # Identify internal vs external
        if carrier_col:
            df_orders['is_external'] = df_orders[carrier_col] >= 9000

        # Monthly aggregation
        monthly_data = []

        for (year, month), group in df_orders.groupby(['year', 'month']):
            record = {
                'date': pd.Timestamp(year=year, month=month, day=1),
                'year': year,
                'month': month,
                'total_orders': len(group)
            }

            # KM billed
            if km_billed_col:
                record['total_km_billed'] = group[km_billed_col].sum()

            # Revenue
            if revenue_col:
                record['revenue_total'] = group[revenue_col].sum()

            # Driver counts
            if carrier_col:
                record['total_drivers'] = group[carrier_col].nunique()
                if 'is_external' in group.columns:
                    record['external_drivers'] = group[group['is_external']][carrier_col].nunique()

            monthly_data.append(record)

        df_monthly = pd.DataFrame(monthly_data)

        # Add tour-based metrics if available
        if len(df_tours) > 0:
            df_monthly = self._add_tour_metrics(df_monthly, df_tours)

        # Sort by date
        df_monthly = df_monthly.sort_values('date').reset_index(drop=True)

        logger.info(f"Created monthly time series: {len(df_monthly)} months")
        logger.info(f"  Date range: {df_monthly['date'].min()} to {df_monthly['date'].max()}")

        return df_monthly

    def _add_tour_metrics(self, df_monthly: pd.DataFrame, df_tours: pd.DataFrame) -> pd.DataFrame:
        """Add tour-based metrics (actual KM, vehicle costs)"""
        logger.info("Adding tour-based metrics...")

        # Find date column in tours
        date_col = self._find_date_column(df_tours)
        if date_col:
            df_tours['date'] = self._convert_dates(df_tours[date_col])
            df_tours['year'] = df_tours['date'].dt.year
            df_tours['month'] = df_tours['date'].dt.month
        else:
            logger.warning("No date column in tours - using source year")
            df_tours['year'] = df_tours['_source_year']
            df_tours['month'] = 1  # Fallback

        # Find relevant columns
        km_actual_col = self._find_column(df_tours, ['IST KM', 'IST.KM', 'Ist_KM'])
        km_cost_col = self._find_column(df_tours, ['PC KM Kosten', 'KM_Kosten'])
        time_col = self._find_column(df_tours, ['IST Zeit PraCar', 'Zeit_PraCar'])
        time_cost_col = self._find_column(df_tours, ['PC Minuten Kosten', 'Minuten_Kosten'])

        # Aggregate tours by month
        tour_monthly = []
        for (year, month), group in df_tours.groupby(['year', 'month']):
            record = {
                'year': year,
                'month': month,
                'total_tours': len(group)
            }

            if km_actual_col:
                record['total_km_actual'] = group[km_actual_col].sum()

            # Calculate vehicle costs
            if km_actual_col and km_cost_col:
                record['vehicle_km_cost'] = (group[km_actual_col] * group[km_cost_col]).sum()

            if time_col and time_cost_col:
                # Time is in hours, cost is per minute
                record['vehicle_time_cost'] = (group[time_col] * 60 * group[time_cost_col]).sum()

            if 'vehicle_km_cost' in record or 'vehicle_time_cost' in record:
                record['total_vehicle_cost'] = record.get('vehicle_km_cost', 0) + record.get('vehicle_time_cost', 0)

            tour_monthly.append(record)

        df_tour_monthly = pd.DataFrame(tour_monthly)

        # Merge with order monthly data
        df_monthly = df_monthly.merge(
            df_tour_monthly,
            on=['year', 'month'],
            how='left'
        )

        return df_monthly

    def _add_working_days(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add working days from external file"""
        logger.info("Adding working days...")

        working_days_file = self.raw_path / 'TRAVECO_Arbeitstage_2022-laufend_für gopf.com_hb v2.xlsx'

        if not working_days_file.exists():
            # Try v1
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

    def _find_date_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the date column in dataframe"""
        candidates = ['Datum.Auftrag', 'Datum', 'Date', 'Tourdatum']
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def _find_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Find a column from list of candidates"""
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def _convert_dates(self, series: pd.Series) -> pd.Series:
        """Convert various date formats to datetime"""
        if pd.api.types.is_datetime64_any_dtype(series):
            return series

        # Try Excel serial dates (numeric)
        if pd.api.types.is_numeric_dtype(series):
            excel_epoch = pd.Timestamp('1899-12-30')
            return excel_epoch + pd.to_timedelta(series, unit='D')

        # Try string formats
        try:
            return pd.to_datetime(series)
        except:
            return pd.to_datetime(series, errors='coerce')


def run():
    """Entry point for pipeline"""
    prep = OperationalDataPrep()
    return prep.run()


if __name__ == '__main__':
    run()
