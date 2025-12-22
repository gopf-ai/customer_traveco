"""Data loaders for Traveco forecasting system"""

import pandas as pd
from pathlib import Path
from typing import Optional, List
import os
import re
from src.utils.config import ConfigLoader
from src.utils.logging_config import get_logger

# Rich components for progress tracking
try:
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
    from rich.console import Console
    RICH_AVAILABLE = True
    rich_console = Console()
except ImportError:
    RICH_AVAILABLE = False
    rich_console = None


logger = get_logger(__name__)


class TravecomDataLoader:
    """
    Load and manage Traveco data files

    Handles loading of:
    - Order analysis (XLSB format)
    - Tour assignments (XLSX format)
    - Customer divisions / Sparten (XLSX)
    - Betriebszentralen mapping (CSV)
    - Working days (XLSX)
    - Personnel costs (Excel) - NEW
    - Total revenue (Excel) - NEW
    """

    def __init__(self, config: Optional[ConfigLoader] = None):
        """
        Initialize data loader

        Args:
            config: Configuration loader instance
        """
        self.config = config if config else ConfigLoader()
        self.data_path = self.config.get_path('data.raw_path')
        logger.info(f"Data loader initialized with path: {self.data_path}")

    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove trailing dots from column names

        Excel exports often have trailing dots (e.g., 'RKdNr.' → 'RKdNr')
        that can cause issues with column references.

        Args:
            df: DataFrame with potentially messy column names

        Returns:
            DataFrame with cleaned column names
        """
        original_cols = df.columns.tolist()
        cleaned_cols = [col.rstrip('.').strip() for col in df.columns]

        # Check for duplicates after cleaning
        if len(cleaned_cols) != len(set(cleaned_cols)):
            logger.warning("Column name cleaning would create duplicates, keeping originals")
            return df

        df.columns = cleaned_cols
        logger.debug(f"Cleaned {len([c for c, o in zip(cleaned_cols, original_cols) if c != o])} column names")

        return df

    def load_order_analysis(self, file_name: Optional[str] = None) -> pd.DataFrame:
        """
        Load main order analysis file (XLSB format)

        Args:
            file_name: Override default file name from config

        Returns:
            DataFrame with order data
        """
        if file_name is None:
            file_name = self.config.get('data.order_analysis')

        file_path = self.data_path / file_name
        logger.info(f"Loading order analysis from: {file_path}")

        if not file_path.exists():
            raise FileNotFoundError(f"Order analysis file not found: {file_path}")

        df = pd.read_excel(file_path, engine='pyxlsb')
        df = self.clean_column_names(df)

        logger.info(f"Loaded {len(df):,} orders with {len(df.columns)} columns")

        return df

    def load_tour_assignments(self, file_name: Optional[str] = None) -> pd.DataFrame:
        """
        Load tour assignments file (XLSX)

        Args:
            file_name: Override default file name from config

        Returns:
            DataFrame with tour data
        """
        if file_name is None:
            file_name = self.config.get('data.tour_assignments')

        file_path = self.data_path / file_name
        logger.info(f"Loading tour assignments from: {file_path}")

        if not file_path.exists():
            raise FileNotFoundError(f"Tour assignments file not found: {file_path}")

        df = pd.read_excel(file_path)

        logger.info(f"Loaded {len(df):,} tours with {len(df.columns)} columns")

        return df

    def load_divisions(self, file_name: Optional[str] = None) -> pd.DataFrame:
        """
        Load customer divisions (Sparten) mapping

        Args:
            file_name: Override default file name from config

        Returns:
            DataFrame with Sparten mapping
        """
        if file_name is None:
            file_name = self.config.get('data.divisions')

        file_path = self.data_path / file_name
        logger.info(f"Loading divisions from: {file_path}")

        if not file_path.exists():
            raise FileNotFoundError(f"Divisions file not found: {file_path}")

        df = pd.read_excel(file_path)

        logger.info(f"Loaded {len(df):,} division mappings")

        return df

    def load_betriebszentralen(self, file_name: Optional[str] = None) -> pd.DataFrame:
        """
        Load Betriebszentralen (14 dispatch centers) mapping

        Args:
            file_name: Override default file name from config

        Returns:
            DataFrame with Betriebszentralen mapping
        """
        if file_name is None:
            file_name = self.config.get('data.betriebszentralen', 'TRAVECO_Betriebszentralen.csv')

        # Try multiple locations
        search_paths = [
            self.data_path / file_name,
            self.data_path.parent / 'raw' / file_name,
            Path('data/raw') / file_name
        ]

        for file_path in search_paths:
            if file_path.exists():
                logger.info(f"Loading Betriebszentralen from: {file_path}")
                df = pd.read_csv(file_path)
                logger.info(f"Loaded {len(df)} Betriebszentralen mappings")
                return df

        raise FileNotFoundError(
            f"Betriebszentralen file not found in any of: {[str(p) for p in search_paths]}"
        )

    def load_working_days(self, file_name: Optional[str] = None) -> pd.DataFrame:
        """
        Load working days data

        Transforms from wide format (Year × 12 month columns) to long format
        (year, month, working_days)

        Args:
            file_name: Override default file name from config

        Returns:
            DataFrame with columns [year, month, working_days]
        """
        if file_name is None:
            file_name = self.config.get(
                'data.working_days',
                'TRAVECO_Arbeitstage_2022-laufend_für gopf.com_hb v1.xlsx'
            )

        file_path = self.data_path / file_name
        logger.info(f"Loading working days from: {file_path}")

        if not file_path.exists():
            logger.warning(f"Working days file not found: {file_path}")
            return pd.DataFrame()

        df = pd.read_excel(file_path)

        # Transform from wide to long format
        # Expected columns: Jahr, Januar, Februar, ..., Dezember
        month_mapping = {
            'Januar': 1, 'Februar': 2, 'März': 3, 'April': 4,
            'Mai': 5, 'Juni': 6, 'Juli': 7, 'August': 8,
            'September': 9, 'Oktober': 10, 'November': 11, 'Dezember': 12
        }

        records = []
        for _, row in df.iterrows():
            year = row['Jahr']
            for month_name, month_num in month_mapping.items():
                if month_name in row and pd.notna(row[month_name]):
                    records.append({
                        'year': int(year),
                        'month': month_num,
                        'working_days': int(row[month_name])
                    })

        df_long = pd.DataFrame(records)

        logger.info(f"Loaded {len(df_long)} working days records ({df_long['year'].min()}-{df_long['year'].max()})")

        return df_long

    def load_personnel_costs(self, file_name: Optional[str] = None) -> pd.DataFrame:
        """
        Load personnel costs data (NEW)

        Expected format: Excel file with columns [date, personnel_costs]
        or [year, month, personnel_costs]

        Args:
            file_name: Override default file name from config

        Returns:
            DataFrame with columns [date, personnel_costs]
        """
        if file_name is None:
            file_name = self.config.get('data.personnel_costs_file')

        if file_name is None:
            logger.debug("Personnel costs file not configured (optional)")
            return pd.DataFrame()

        file_path = self.data_path / file_name
        logger.debug(f"Loading personnel costs from: {file_path}")

        if not file_path.exists():
            logger.debug(f"Personnel costs file not found: {file_path} (optional)")
            return pd.DataFrame()

        df = pd.read_excel(file_path)

        # Standardize to date format
        if 'date' not in df.columns and 'year' in df.columns and 'month' in df.columns:
            df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
            df = df[['date', 'personnel_costs']]

        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        else:
            raise ValueError("Personnel costs file must have either 'date' or 'year'+'month' columns")

        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)

        logger.info(f"Loaded {len(df)} personnel cost records ({df['date'].min()} to {df['date'].max()})")

        return df

    def load_total_revenue(self, file_name: Optional[str] = None) -> pd.DataFrame:
        """
        Load total revenue data (transportation + non-transportation) (NEW)

        Expected format: Excel file with columns [date, total_revenue_all]
        or [year, month, total_revenue_all]

        Args:
            file_name: Override default file name from config

        Returns:
            DataFrame with columns [date, total_revenue_all]
        """
        if file_name is None:
            file_name = self.config.get('data.total_revenue_file')

        if file_name is None:
            logger.debug("Total revenue file not configured (optional)")
            return pd.DataFrame()

        file_path = self.data_path / file_name
        logger.debug(f"Loading total revenue from: {file_path}")

        if not file_path.exists():
            logger.debug(f"Total revenue file not found: {file_path} (optional)")
            return pd.DataFrame()

        df = pd.read_excel(file_path)

        # Standardize to date format
        if 'date' not in df.columns and 'year' in df.columns and 'month' in df.columns:
            df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
            df = df[['date', 'total_revenue_all']]

        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        else:
            raise ValueError("Total revenue file must have either 'date' or 'year'+'month' columns")

        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)

        logger.info(f"Loaded {len(df)} total revenue records ({df['date'].min()} to {df['date'].max()})")

        return df

    def load_historic_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Load consolidated historic order data (2022-2024)

        Args:
            start_date: Filter start date (YYYY-MM-DD)
            end_date: Filter end date (YYYY-MM-DD)

        Returns:
            DataFrame with historic order data
        """
        file_path = self.config.get_path('data.processed_path') / 'historic_orders_2022_2024.parquet'

        logger.info(f"Loading historic data from: {file_path}")

        if not file_path.exists():
            logger.warning(f"Historic data file not found: {file_path}")
            logger.warning("Will need to aggregate from individual monthly files")
            return pd.DataFrame()

        df = pd.read_parquet(file_path)

        # Apply date filter
        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]

        logger.info(f"Loaded {len(df):,} historic orders ({df['date'].min()} to {df['date'].max()})")

        return df

    def load_historic_orders_multi_year(
        self,
        years: List[int] = None,
        base_path: Path = None
    ) -> pd.DataFrame:
        """
        Load and combine all monthly order files across multiple years

        Scans data/raw/{year}/ directories for monthly Excel files and combines them.

        Args:
            years: List of years to load (default: [2022, 2023, 2024, 2025])
            base_path: Base directory containing year folders (default: data/raw)

        Returns:
            DataFrame with combined order data from all years
        """
        if years is None:
            years = self.config.get('data.historic_years', [2022, 2023, 2024, 2025])

        if base_path is None:
            # Use data/raw instead of the swisstransfer folder
            base_path = Path(self.config.get_path('data.processed_path')).parent / 'raw'

        logger.info(f"Loading historic orders from {len(years)} years: {years}")
        logger.info(f"Base path: {base_path}")

        all_dataframes = []
        total_files = 0

        # First pass: count files
        for year in years:
            year_path = base_path / str(year)
            if not year_path.exists():
                logger.warning(f"Year directory not found: {year_path}")
                continue

            files = [f for f in os.listdir(year_path)
                    if 'QS Auftragsanalyse' in f and f.endswith(('.xlsx', '.xlsb'))]
            total_files += len(files)

        logger.info(f"Found {total_files} monthly files to load")

        # Second pass: load files with progress tracking
        if RICH_AVAILABLE and rich_console:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("({task.completed}/{task.total})"),
                TimeElapsedColumn(),
                console=rich_console
            ) as progress:
                task = progress.add_task("[cyan]Loading monthly files...", total=total_files)

                for year in years:
                    year_path = base_path / str(year)
                    if not year_path.exists():
                        continue

                    # Find all monthly files
                    files = sorted([f for f in os.listdir(year_path)
                                   if 'QS Auftragsanalyse' in f and f.endswith(('.xlsx', '.xlsb'))])

                    for file_name in files:
                        file_path = year_path / file_name
                        progress.update(task, description=f"[cyan]Loading {year}/{file_name[:20]}...")

                        try:
                            # Load Excel file
                            if file_name.endswith('.xlsb'):
                                df = pd.read_excel(file_path, engine='pyxlsb')
                            else:
                                df = pd.read_excel(file_path)

                            # Clean column names
                            df = self.clean_column_names(df)

                            # Add source metadata
                            df['_source_year'] = year
                            df['_source_file'] = file_name

                            all_dataframes.append(df)
                            progress.advance(task)

                        except Exception as e:
                            logger.error(f"Error loading {file_path}: {e}")
                            progress.advance(task)
                            continue
        else:
            # Fallback without progress bar
            file_count = 0
            for year in years:
                year_path = base_path / str(year)
                if not year_path.exists():
                    continue

                files = sorted([f for f in os.listdir(year_path)
                               if 'QS Auftragsanalyse' in f and f.endswith(('.xlsx', '.xlsb'))])

                for file_name in files:
                    file_count += 1
                    file_path = year_path / file_name
                    logger.info(f"  [{file_count}/{total_files}] Loading {year}/{file_name}")

                    try:
                        if file_name.endswith('.xlsb'):
                            df = pd.read_excel(file_path, engine='pyxlsb')
                        else:
                            df = pd.read_excel(file_path)

                        df = self.clean_column_names(df)
                        df['_source_year'] = year
                        df['_source_file'] = file_name
                        all_dataframes.append(df)

                    except Exception as e:
                        logger.error(f"Error loading {file_path}: {e}")
                        continue

        # Combine all dataframes
        if not all_dataframes:
            raise ValueError(f"No data files found in {base_path} for years {years}")

        logger.info(f"Combining {len(all_dataframes)} monthly files...")
        df_combined = pd.concat(all_dataframes, ignore_index=True)

        logger.info(f"✓ Loaded {len(df_combined):,} orders from {len(all_dataframes)} files")
        logger.info(f"  Columns: {len(df_combined.columns)}")
        logger.info(f"  Years: {sorted(df_combined['_source_year'].unique().tolist())}")

        return df_combined

    def load_historic_tours_multi_year(
        self,
        years: List[int] = None,
        base_path: Path = None
    ) -> pd.DataFrame:
        """
        Load and combine all tour files across multiple years

        Args:
            years: List of years to load (default: [2022, 2023, 2024, 2025])
            base_path: Base directory containing year folders (default: data/raw)

        Returns:
            DataFrame with combined tour data from all years
        """
        if years is None:
            years = self.config.get('data.historic_years', [2022, 2023, 2024, 2025])

        if base_path is None:
            base_path = Path(self.config.get_path('data.processed_path')).parent / 'raw'

        logger.info(f"Loading historic tours from {len(years)} years: {years}")

        all_dataframes = []

        for year in years:
            year_path = base_path / str(year)
            if not year_path.exists():
                logger.warning(f"Year directory not found: {year_path}")
                continue

            # Find tour file (pattern: "QS Tourenaufstellung")
            tour_files = [f for f in os.listdir(year_path)
                         if 'QS Tourenaufstellung' in f and f.endswith('.xlsx')]

            if not tour_files:
                logger.warning(f"No tour file found for year {year}")
                continue

            # Load first matching file
            file_path = year_path / tour_files[0]
            logger.info(f"  Loading {year}/{tour_files[0]}")

            try:
                df = pd.read_excel(file_path)
                df = self.clean_column_names(df)
                df['_source_year'] = year
                all_dataframes.append(df)

            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue

        if not all_dataframes:
            logger.warning(f"No tour files found for years {years}")
            return pd.DataFrame()

        # Combine all dataframes
        df_combined = pd.concat(all_dataframes, ignore_index=True)

        logger.info(f"✓ Loaded {len(df_combined):,} tours from {len(all_dataframes)} files")

        return df_combined

    def load_personnel_costs_single_year(self, year: int) -> pd.DataFrame:
        """
        Load personnel costs for a single year.

        Extracts 4 key personnel metrics from Excel files:
        - Row 8: Saldo Mehrarbeitszeit h (Overtime balance in hours)
        - Row 10: Feriensaldo t (Vacation balance in days)
        - Row 16: Krank h (Sick leave in hours)
        - Row 18: Total Ges. Absenz (Total absence in hours)

        Args:
            year: Year to load (2022-2025)

        Returns:
            DataFrame with columns: metric_name, jan, feb, mar, ..., dec
        """
        # Build file path
        file_path = Path(f'data/raw/{year}/{year} Personal/Personal {year}.xlsx')

        if not file_path.exists():
            logger.warning(f"Personnel file not found: {file_path}")
            return pd.DataFrame()

        logger.info(f"Loading personnel costs from {file_path}")

        # Read Excel file
        # Row 6: Month headers (Jan, Feb, Mär, ...)
        # Row 7: Unit indicator (TFr.)
        # Row 8+: Data starts
        df = pd.read_excel(
            file_path,
            sheet_name=str(year),
            skiprows=7,  # Skip to row 8 (data starts)
            usecols='A,C:N',  # Column A (metric name) + C-N (months Jan-Dec)
            nrows=12,  # Read 12 potential rows
            engine='openpyxl'
        )

        # Set proper column names
        month_names = ['jan', 'feb', 'mar', 'apr', 'mai', 'jun',
                      'jul', 'aug', 'sep', 'okt', 'nov', 'dez']
        df.columns = ['metric_name'] + month_names

        # Define target metrics and their row indices (0-indexed after skiprows)
        # Row 8 in Excel → index 0 after skiprows=7
        # Row 10 in Excel → index 2 after skiprows=7
        # Row 16 in Excel → index 8 after skiprows=7
        # Row 18 in Excel → index 10 after skiprows=7
        target_metrics = {
            0: 'Saldo Mehrarbeitszeit h',
            2: 'Feriensaldo t',
            8: 'Krank h',
            10: 'Total Ges. Absenz'
        }

        # Filter to only target rows
        df_filtered = []
        for idx, metric_name in target_metrics.items():
            if idx < len(df):
                row = df.iloc[idx:idx+1].copy()
                # Override metric name to standardize
                row['metric_name'] = metric_name
                df_filtered.append(row)

        if not df_filtered:
            logger.warning(f"No personnel metrics found in {file_path}")
            return pd.DataFrame()

        df_result = pd.concat(df_filtered, ignore_index=True)

        # For 2025: only 9 months (Jan-Sep), set Oct-Dec to NaN
        if year == 2025:
            for month in ['okt', 'nov', 'dez']:
                if month in df_result.columns:
                    df_result[month] = None

        logger.info(f"✓ Loaded {len(df_result)} metrics for {year}")

        return df_result

    def load_personnel_costs_multi_year(self, years: List[int] = None) -> pd.DataFrame:
        """
        Load personnel costs for multiple years and pivot to wide format.

        Creates a wide-format DataFrame with:
        - Rows: 4 metrics
        - Columns: {year}_{month} for all year-month combinations

        Args:
            years: List of years to load (default: [2022, 2023, 2024, 2025])

        Returns:
            Wide-format DataFrame with columns like: metric_name, 2022_jan, 2022_feb, ...
        """
        if years is None:
            years = [2022, 2023, 2024, 2025]

        logger.info(f"Loading personnel costs for years: {years}")

        all_data = []

        # Load each year
        for year in years:
            df_year = self.load_personnel_costs_single_year(year)

            if df_year.empty:
                logger.warning(f"Skipping {year} - no data loaded")
                continue

            # Add year prefix to month columns
            month_cols = [col for col in df_year.columns if col != 'metric_name']
            rename_map = {col: f'{year}_{col}' for col in month_cols}
            df_year = df_year.rename(columns=rename_map)

            all_data.append(df_year)

        if not all_data:
            logger.error("No personnel data loaded for any year")
            return pd.DataFrame()

        # Merge all years on metric_name
        df_result = all_data[0]
        for df in all_data[1:]:
            df_result = df_result.merge(df, on='metric_name', how='outer')

        # Reorder columns: metric_name, then chronological year_month
        month_order = ['jan', 'feb', 'mar', 'apr', 'mai', 'jun',
                      'jul', 'aug', 'sep', 'okt', 'nov', 'dez']

        ordered_cols = ['metric_name']
        for year in sorted(years):
            for month in month_order:
                col_name = f'{year}_{month}'
                if col_name in df_result.columns:
                    ordered_cols.append(col_name)

        df_result = df_result[ordered_cols]

        # Calculate total months with data (excluding 2025 Q4)
        data_cols = [col for col in df_result.columns if col != 'metric_name']
        total_months = len([col for col in data_cols if not col.startswith('2025_') or
                          col.endswith(('jan', 'feb', 'mar', 'apr', 'mai', 'jun', 'jul', 'aug', 'sep'))])

        logger.info(f"✓ Combined personnel costs: {len(df_result)} metrics × {total_months} months")

        return df_result

    # Aliases for pipeline compatibility
    def load_orders(self, file_name: Optional[str] = None) -> pd.DataFrame:
        """Alias for load_order_analysis()"""
        return self.load_order_analysis(file_name)

    def load_tours(self, file_name: Optional[str] = None) -> pd.DataFrame:
        """Alias for load_tour_assignments()"""
        return self.load_tour_assignments(file_name)
