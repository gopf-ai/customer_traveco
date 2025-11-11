"""Data loaders for Traveco forecasting system"""

import pandas as pd
from pathlib import Path
from typing import Optional
from src.utils.config import ConfigLoader
from src.utils.logging_config import get_logger


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
            logger.warning("Personnel costs file not configured")
            return pd.DataFrame()

        file_path = self.data_path / file_name
        logger.info(f"Loading personnel costs from: {file_path}")

        if not file_path.exists():
            logger.warning(f"Personnel costs file not found: {file_path}")
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
            logger.warning("Total revenue file not configured")
            return pd.DataFrame()

        file_path = self.data_path / file_name
        logger.info(f"Loading total revenue from: {file_path}")

        if not file_path.exists():
            logger.warning(f"Total revenue file not found: {file_path}")
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

    # Aliases for pipeline compatibility
    def load_orders(self, file_name: Optional[str] = None) -> pd.DataFrame:
        """Alias for load_order_analysis()"""
        return self.load_order_analysis(file_name)

    def load_tours(self, file_name: Optional[str] = None) -> pd.DataFrame:
        """Alias for load_tour_assignments()"""
        return self.load_tour_assignments(file_name)
