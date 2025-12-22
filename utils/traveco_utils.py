"""
Traveco Forecasting Utilities

This module provides reusable classes and functions for the Traveco transport
forecasting project. It includes data loading, feature engineering, model training,
and evaluation utilities.

Author: Kevin Kuhn
Date: 2025-10-17
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from typing import List, Dict, Tuple, Optional
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')


class ConfigLoader:
    """Load and manage project configuration"""

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize configuration loader

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> dict:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config

    def get(self, key: str, default=None):
        """Get configuration value by key (supports dot notation)"""
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value


class TravecomDataLoader:
    """Load and validate Traveco data files"""

    def __init__(self, config: Optional[ConfigLoader] = None):
        """
        Initialize data loader

        Args:
            config: ConfigLoader instance, creates new one if None
        """
        self.config = config if config else ConfigLoader()
        self.data_path = Path(self.config.get('data.raw_path'))

    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean column names by removing trailing dots and extra spaces

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with cleaned column names
        """
        # Store original names for reference
        original_cols = df.columns.tolist()

        # Clean column names: remove trailing dots and strip spaces
        cleaned_cols = [col.rstrip('.').strip() if isinstance(col, str) else col for col in df.columns]

        # Check for duplicates after cleaning
        if len(cleaned_cols) != len(set(cleaned_cols)):
            # If duplicates, keep original names
            print("   ⚠️  Column name cleaning would create duplicates - keeping original names")
            return df

        # Apply cleaned names
        df.columns = cleaned_cols

        # Show what was changed
        changes = [(orig, clean) for orig, clean in zip(original_cols, cleaned_cols) if orig != clean]
        if changes:
            print(f"   ✓ Cleaned {len(changes)} column names (removed trailing dots)")
            for orig, clean in changes[:5]:  # Show first 5
                print(f"      '{orig}' → '{clean}'")
            if len(changes) > 5:
                print(f"      ... and {len(changes)-5} more")

        return df

    def load_order_analysis(self) -> pd.DataFrame:
        """
        Load main order analysis file (Auftragsanalyse)

        Returns:
            DataFrame with order data
        """
        file_name = self.config.get('data.order_analysis')
        file_path = self.data_path / file_name

        print(f"Loading order analysis from: {file_path}")

        if not file_path.exists():
            raise FileNotFoundError(f"Order analysis file not found: {file_path}")

        # Read Excel file (xlsb format)
        df = pd.read_excel(file_path, engine='pyxlsb')

        print(f"Loaded {len(df):,} orders with {len(df.columns)} columns")

        # Clean column names (remove trailing dots like "RKdNr.")
        df = self.clean_column_names(df)

        return df

    def load_tour_assignments(self) -> pd.DataFrame:
        """
        Load tour assignments file (Tourenaufstellung)

        Returns:
            DataFrame with tour assignment data
        """
        file_name = self.config.get('data.tour_assignments')
        file_path = self.data_path / file_name

        print(f"Loading tour assignments from: {file_path}")

        if not file_path.exists():
            raise FileNotFoundError(f"Tour assignments file not found: {file_path}")

        df = pd.read_excel(file_path)

        print(f"Loaded {len(df):,} tour assignments")

        return df

    def load_divisions(self) -> pd.DataFrame:
        """
        Load customer divisions file (Sparten)

        Returns:
            DataFrame with customer division mapping
        """
        file_name = self.config.get('data.divisions')
        file_path = self.data_path / file_name

        print(f"Loading divisions from: {file_path}")

        if not file_path.exists():
            raise FileNotFoundError(f"Divisions file not found: {file_path}")

        df = pd.read_excel(file_path)

        print(f"Loaded {len(df):,} customer division mappings")

        return df

    def load_betriebszentralen(self) -> pd.DataFrame:
        """
        Load Betriebszentralen (dispatch centers) mapping file

        Returns:
            DataFrame with Betriebszentralen mapping (14 invoicing units)
        """
        file_name = self.config.get('data.betriebszentralen', 'TRAVECO_Betriebszentralen.csv')

        # Try raw path first
        file_path = self.data_path / file_name

        # If not found, try data/raw/ subdirectory
        if not file_path.exists():
            file_path = self.data_path.parent / 'raw' / file_name

        print(f"Loading Betriebszentralen from: {file_path}")

        if not file_path.exists():
            raise FileNotFoundError(f"Betriebszentralen file not found: {file_path}")

        df = pd.read_csv(file_path)

        print(f"Loaded {len(df):,} Betriebszentralen (dispatch center) mappings")

        return df

    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all data files

        Returns:
            Tuple of (orders, tours, divisions) DataFrames
        """
        orders = self.load_order_analysis()
        tours = self.load_tour_assignments()
        divisions = self.load_divisions()

        return orders, tours, divisions

    def load_all_with_betriebszentralen(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all data files including Betriebszentralen

        Returns:
            Tuple of (orders, tours, divisions, betriebszentralen) DataFrames
        """
        orders = self.load_order_analysis()
        tours = self.load_tour_assignments()
        divisions = self.load_divisions()
        betriebszentralen = self.load_betriebszentralen()

        return orders, tours, divisions, betriebszentralen


class TravecomFeatureEngine:
    """Feature engineering utilities for Traveco data"""

    def __init__(self, config: Optional[ConfigLoader] = None):
        """
        Initialize feature engine

        Args:
            config: ConfigLoader instance
        """
        self.config = config if config else ConfigLoader()

    def convert_date_column(self, date_column: pd.Series) -> pd.Series:
        """
        Convert date column to proper datetime

        Handles:
        - Excel serial dates (numeric: 45809 = June 1, 2025)
        - ISO format strings (YYYY-MM-DD from CSV)
        - Swiss date format strings (DD.MM.YYYY)
        - Already converted datetime

        Args:
            date_column: Pandas Series with dates

        Returns:
            Series with proper datetime values
        """
        # If already datetime, return as-is
        if pd.api.types.is_datetime64_any_dtype(date_column):
            return date_column

        # Check if it's numeric (Excel serial date)
        if pd.api.types.is_numeric_dtype(date_column):
            # Excel serial dates: days since 1899-12-30 (Excel's epoch)
            excel_epoch = pd.Timestamp('1899-12-30')
            result = pd.to_datetime(excel_epoch) + pd.to_timedelta(date_column, unit='D')
            return result

        # Try different string formats
        # First, try ISO format (from CSV files)
        try:
            result = pd.to_datetime(date_column, format='ISO8601')
            return result
        except:
            pass

        # Try Swiss format (DD.MM.YYYY)
        try:
            result = pd.to_datetime(date_column, format='%d.%m.%Y', dayfirst=True)
            return result
        except:
            pass

        # Final fallback: let pandas infer
        try:
            result = pd.to_datetime(date_column)
            return result
        except Exception as e:
            raise ValueError(f"Could not convert date column. Error: {e}")

    def extract_temporal_features(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """
        Extract temporal features from date column

        Args:
            df: Input DataFrame
            date_column: Name of date column

        Returns:
            DataFrame with added temporal features
        """
        df = df.copy()

        # Convert to datetime using smart conversion
        df[date_column] = self.convert_date_column(df[date_column])

        # Extract temporal features
        temporal_features = self.config.get('features.temporal_features', [
            'year', 'month', 'week', 'quarter', 'day_of_year', 'weekday'
        ])

        if 'year' in temporal_features:
            df['year'] = df[date_column].dt.year

        if 'month' in temporal_features:
            df['month'] = df[date_column].dt.month

        if 'week' in temporal_features:
            df['week'] = df[date_column].dt.isocalendar().week

        if 'quarter' in temporal_features:
            df['quarter'] = df[date_column].dt.quarter

        if 'day_of_year' in temporal_features:
            df['day_of_year'] = df[date_column].dt.dayofyear

        if 'weekday' in temporal_features:
            df['weekday'] = df[date_column].dt.dayofweek

        print(f"Extracted {len(temporal_features)} temporal features")

        return df

    def create_lag_features(self, df: pd.DataFrame, target_col: str,
                           group_col: Optional[str] = None) -> pd.DataFrame:
        """
        Create lag features for time series forecasting

        Args:
            df: Input DataFrame (must be sorted by date)
            target_col: Target column to create lags for
            group_col: Optional grouping column (e.g., 'branch')

        Returns:
            DataFrame with added lag features
        """
        df = df.copy()

        lag_periods = self.config.get('features.lag_periods', [1, 3, 6, 12])

        for lag in lag_periods:
            col_name = f'{target_col}_lag_{lag}'

            if group_col:
                df[col_name] = df.groupby(group_col)[target_col].shift(lag)
            else:
                df[col_name] = df[target_col].shift(lag)

        print(f"Created {len(lag_periods)} lag features for {target_col}")

        return df

    def calculate_time_weights(self, dates: pd.Series, decay_rate: float = 0.3) -> np.ndarray:
        """
        Calculate exponential time decay weights

        Formula: w(t) = exp(-λ * (T - t) / 365)
        where λ is the decay rate

        Args:
            dates: Series of dates
            decay_rate: Decay rate (λ), default 0.3

        Returns:
            Array of weights (normalized)
        """
        dates = pd.to_datetime(dates)

        # Calculate days from most recent date
        days_from_recent = (dates.max() - dates).dt.days

        # Calculate exponential decay weights
        weights = np.exp(-decay_rate * days_from_recent / 365)

        # Normalize weights (sum to 1)
        weights_normalized = weights / weights.sum()

        # Scale back to original length for data duplication
        weights_scaled = weights_normalized * len(weights)

        return weights_scaled

    def identify_carrier_type(self, df: pd.DataFrame, carrier_col: str = 'Nummer.Spedition') -> pd.DataFrame:
        """
        Identify internal vs external carriers

        Args:
            df: Input DataFrame
            carrier_col: Name of carrier number column

        Returns:
            DataFrame with added 'carrier_type' column
        """
        df = df.copy()

        internal_max = self.config.get('filtering.internal_carrier_max', 8889)
        external_min = self.config.get('filtering.external_carrier_min', 9000)

        # Convert carrier column to numeric (may be string from Excel)
        df[carrier_col] = pd.to_numeric(df[carrier_col], errors='coerce')

        df['carrier_type'] = df[carrier_col].apply(
            lambda x: 'internal' if pd.notna(x) and x <= internal_max
                     else 'external' if pd.notna(x) and x >= external_min
                     else 'unknown'
        )

        print(f"Carrier type distribution:\n{df['carrier_type'].value_counts()}")

        return df

    def classify_order_type_multifield(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify order types using multi-field logic (CORRECTED per Christian's feedback)

        Uses combination of:
        - Column K: Auftragsart (Order type)
        - Column AU: Lieferart 2.0 (Delivery type)
        - Column CW: System_id.Auftrag (System ID)

        Classification logic from Christian's email (October 2025):
        1. B&T Fossil: AU='B&T Fossil' + CW='B&T'
        2. B&T Holzpellets: AU='B&T Holzpellets' + CW='B&T'
        3. Flüssigtransporte: AU='Flüssigtransporte' + CW='TRP'
        4. Palettentransporte - Leergut: K='Leergut' + AU='Palettentransporte' + CW='TRP'
        5. Palettentransporte - Lieferung: K='Lieferung' + AU='Palettentransporte' + CW='TRP'
        6. Palettentransporte - Retoure: K in ['Retoure','Abholung'] + AU='Palettentransporte' + CW='TRP'
        7. EXCLUDE: Losetransporte (contract weight issues)

        Args:
            df: Input DataFrame with K, AU, CW columns

        Returns:
            DataFrame with added 'order_type_detailed' column
        """
        df = df.copy()

        print(f"\n📦 Classifying order types (multi-field logic):")

        # Check required columns (use 'Auftrags-art' with hyphen as in Excel file)
        required_cols = ['Auftrags-art', 'Lieferart 2.0', 'System_id.Auftrag']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"   ⚠️  Missing columns: {missing_cols}")
            print(f"   Returning DataFrame without classification")
            df['order_type_detailed'] = 'Unknown'
            return df

        def classify_row(row):
            k = row['Auftrags-art'] if pd.notna(row['Auftrags-art']) else ''
            au = row['Lieferart 2.0'] if pd.notna(row['Lieferart 2.0']) else ''
            cw = row['System_id.Auftrag'] if pd.notna(row['System_id.Auftrag']) else ''

            # B&T deliveries
            if au == 'B&T Fossil' and cw == 'B&T':
                return 'B&T Fossil Delivery'
            if au == 'B&T Holzpellets' and cw == 'B&T':
                return 'B&T Pellets Delivery'

            # TRP categories
            if au == 'Flüssigtransporte' and cw == 'TRP':
                return 'Liquid Transport'

            # Palettentransporte subcategories
            if au == 'Palettentransporte' and cw == 'TRP':
                if k == 'Leergut':
                    return 'Leergut (Empty Returns)'
                elif k in ['Retoure', 'Abholung']:
                    return 'Retoure (Return/Pickup)'
                elif k == 'Lieferung' or k == '':
                    # Empty Auftragsart defaults to Lieferung for Palettentransporte
                    return 'Pallet Delivery'

            # Losetransporte - mark for exclusion
            if au == 'Losetransporte':
                return 'EXCLUDE - Losetransporte'

            # Default
            return 'Other'

        # Apply classification
        df['order_type_detailed'] = df.apply(classify_row, axis=1)

        # Print distribution
        print(f"   ✓ Order type distribution:")
        print(df['order_type_detailed'].value_counts().to_string())

        # Check for Losetransporte to exclude
        exclude_count = (df['order_type_detailed'] == 'EXCLUDE - Losetransporte').sum()
        if exclude_count > 0:
            print(f"\n   ⚠️  Found {exclude_count:,} Losetransporte orders")
            print(f"      → These should be excluded due to contract weight issues")

        return df

    def map_customer_divisions(self, df_orders: pd.DataFrame, df_divisions: pd.DataFrame,
                              customer_col: str = 'RKdNr',
                              division_col: str = 'Sparte') -> pd.DataFrame:
        """
        Map customer numbers to their divisions (Sparten)

        Args:
            df_orders: Orders DataFrame
            df_divisions: Divisions DataFrame (from Sparten.xlsx)
            customer_col: Customer number column in orders
            division_col: Division column in divisions file

        Returns:
            DataFrame with added 'sparte' column
        """
        df_orders = df_orders.copy()
        df_divisions = df_divisions.copy()

        # Identify the customer number column in divisions file
        # Usually the first column (Kunden-Nr.)
        divisions_customer_col = df_divisions.columns[0]

        print(f"\n🔍 Sparten mapping diagnostics:")
        print(f"   Orders customer column: '{customer_col}'")
        print(f"   Divisions customer column: '{divisions_customer_col}'")
        print(f"   Orders customer type: {df_orders[customer_col].dtype}")
        print(f"   Divisions customer type: {df_divisions[divisions_customer_col].dtype}")

        # Normalize both to same type (convert to Int64 for reliable matching)
        # This handles float (946200.0) vs int (946200) mismatches
        try:
            df_orders[customer_col] = pd.to_numeric(df_orders[customer_col], errors='coerce').astype('Int64')
            df_divisions[divisions_customer_col] = pd.to_numeric(df_divisions[divisions_customer_col], errors='coerce').astype('Int64')
            print(f"   ✓ Converted both to Int64 for matching")
        except Exception as e:
            print(f"   ⚠️  Type conversion failed: {e}")
            print(f"   Proceeding with original types")

        # Check for matches before mapping
        orders_customers = set(df_orders[customer_col].dropna().unique())
        divisions_customers = set(df_divisions[divisions_customer_col].dropna().unique())
        matching = orders_customers & divisions_customers

        print(f"   Unique customers in orders: {len(orders_customers):,}")
        print(f"   Unique customers in divisions: {len(divisions_customers):,}")
        print(f"   Matching customers: {len(matching):,}")

        if len(matching) == 0:
            print(f"\n   ⚠️  WARNING: No matching customers found!")
            print(f"   Sample from orders: {list(orders_customers)[:3]}")
            print(f"   Sample from divisions: {list(divisions_customers)[:3]}")
            print(f"   All orders will be marked as 'Keine Sparte (Traveco)'")

        # Create mapping dictionary
        division_mapping = df_divisions.set_index(divisions_customer_col)[division_col].to_dict()

        # Map to orders
        df_orders['sparte'] = df_orders[customer_col].map(division_mapping)

        # Handle unmapped customers (CORRECTED per Christian's feedback Oct 2025)
        unmapped_count = df_orders['sparte'].isna().sum()
        mapped_count = len(df_orders) - unmapped_count

        if unmapped_count > 0:
            # Check if unmapped orders have TRAVECO as customer name
            if 'RKdName' in df_orders.columns:
                traveco_mask = (df_orders['sparte'].isna()) & (df_orders['RKdName'].str.contains('TRAVECO', case=False, na=False))
                traveco_count = traveco_mask.sum()

                if traveco_count > 0:
                    df_orders.loc[traveco_mask, 'sparte'] = 'TRAVECO Intern'
                    print(f"\n📊 Special handling:")
                    print(f"   ✓ Found {traveco_count:,} orders with TRAVECO as customer → marked as 'TRAVECO Intern'")

            # Remaining unmapped get generic label
            remaining_unmapped = df_orders['sparte'].isna().sum()
            if remaining_unmapped > 0:
                df_orders['sparte'] = df_orders['sparte'].fillna('Keine Sparte')

            print(f"\n📊 Mapping results:")
            print(f"   ✓ Mapped to divisions: {mapped_count:,} orders ({mapped_count/len(df_orders)*100:.1f}%)")
            if 'RKdName' in df_orders.columns and traveco_count > 0:
                print(f"   ✓ TRAVECO Intern: {traveco_count:,} orders ({traveco_count/len(df_orders)*100:.1f}%)")
            if remaining_unmapped > 0:
                print(f"   ⚠️  Unmapped: {remaining_unmapped:,} orders ({remaining_unmapped/len(df_orders)*100:.1f}%)")
                print(f"      → Marked as 'Keine Sparte'")
        else:
            print(f"\n✓ All {len(df_orders):,} orders successfully mapped to Sparten!")

        print(f"\n✓ Sparten mapping complete:")
        print(f"   Total divisions: {df_orders['sparte'].nunique()}")
        print(f"   Top 10 divisions:")
        print(df_orders['sparte'].value_counts().head(10))

        return df_orders

    def map_betriebszentralen(self, df_orders: pd.DataFrame, df_betriebszentralen: pd.DataFrame,
                              auftraggeber_col: str = 'Nummer.Auftraggeber') -> pd.DataFrame:
        """
        Map Auftraggeber numbers to Betriebszentralen (dispatch center) names
        (CORRECTED per Christian's feedback Oct 2025)

        This maps the 13 invoicing units (Betriebszentralen) that are the actual
        dispatch centers where transports originate and invoices are sent.

        **CORRECTION**: BZ 10 and BZ 9000 are duplicates (warehouse relocation from Hägendorf to Nebikon).
        We merge BZ 10 → BZ 9000 (both "LC Nebikon") for accurate analysis.

        Args:
            df_orders: Orders DataFrame
            df_betriebszentralen: Betriebszentralen DataFrame (from TRAVECO_Betriebszentralen.csv)
            auftraggeber_col: Auftraggeber column name in orders

        Returns:
            DataFrame with added 'betriebszentrale_name' column
        """
        df_orders = df_orders.copy()
        df_betriebszentralen = df_betriebszentralen.copy()

        print(f"\n🏢 Betriebszentralen mapping diagnostics (with BZ 10→9000 merge):")
        print(f"   Orders Auftraggeber column: '{auftraggeber_col}'")
        print(f"   Orders Auftraggeber type: {df_orders[auftraggeber_col].dtype}")
        print(f"   Betriebszentralen type: {df_betriebszentralen['Nummer.Auftraggeber'].dtype}")

        # CRITICAL CORRECTION: Merge BZ 10 → 9000 (same location after relocation)
        if auftraggeber_col in df_orders.columns:
            # First, replace placeholder values ('-', '', etc.) with NaN
            placeholder_count = df_orders[auftraggeber_col].isin(['-', '', ' ']).sum()
            if placeholder_count > 0:
                df_orders[auftraggeber_col] = df_orders[auftraggeber_col].replace(['-', '', ' '], pd.NA)
                print(f"\n   ℹ️  Found {placeholder_count:,} orders with placeholder Auftraggeber ('-') → set to NaN")

            bz10_count = (df_orders[auftraggeber_col] == 10).sum()
            if bz10_count > 0:
                df_orders[auftraggeber_col] = df_orders[auftraggeber_col].replace(10, 9000)
                print(f"   ✓ Merged BZ 10 → BZ 9000: {bz10_count:,} orders")
                print(f"      (Both represent LC Nebikon after warehouse relocation)")
            else:
                print(f"   ℹ️  No BZ 10 orders found (already merged or not present)")

        # Normalize both to same type (Int64 for reliable matching)
        try:
            df_orders[auftraggeber_col] = pd.to_numeric(df_orders[auftraggeber_col], errors='coerce').astype('Int64')
            df_betriebszentralen['Nummer.Auftraggeber'] = pd.to_numeric(df_betriebszentralen['Nummer.Auftraggeber'], errors='coerce').astype('Int64')
            print(f"   ✓ Converted both to Int64 for matching")
        except Exception as e:
            print(f"   ⚠️  Type conversion failed: {e}")
            print(f"   Proceeding with original types")

        # Check for matches before mapping
        orders_numbers = set(df_orders[auftraggeber_col].dropna().unique())
        betriebszentralen_numbers = set(df_betriebszentralen['Nummer.Auftraggeber'].dropna().unique())
        matching = orders_numbers & betriebszentralen_numbers

        print(f"   Unique Auftraggeber in orders: {len(orders_numbers):,}")
        print(f"   Unique Betriebszentralen numbers: {len(betriebszentralen_numbers):,}")
        print(f"   Matching numbers: {len(matching):,}")

        if len(matching) == 0:
            print(f"\n   ⚠️  WARNING: No matching Betriebszentralen found!")
            print(f"   Sample from orders: {list(orders_numbers)[:5]}")
            print(f"   Sample from Betriebszentralen: {list(betriebszentralen_numbers)[:5]}")

        # Handle duplicates: use first match (10 and 9000 both = LC Nebikon)
        # Drop duplicates keeping first occurrence
        betriebszentralen_unique = df_betriebszentralen.drop_duplicates(
            subset='Nummer.Auftraggeber', keep='first'
        )

        duplicates_count = len(df_betriebszentralen) - len(betriebszentralen_unique)
        if duplicates_count > 0:
            print(f"\n   ℹ️  Found {duplicates_count} duplicate Auftraggeber numbers (keeping first match)")

        # Create mapping dictionary: Nummer.Auftraggeber -> Name1
        betriebszentralen_mapping = betriebszentralen_unique.set_index('Nummer.Auftraggeber')['Name1'].to_dict()

        # Map to orders
        df_orders['betriebszentrale_name'] = df_orders[auftraggeber_col].map(betriebszentralen_mapping)

        # Handle unmapped (mark as "Unknown Betriebszentrale")
        unmapped_count = df_orders['betriebszentrale_name'].isna().sum()
        mapped_count = len(df_orders) - unmapped_count

        if unmapped_count > 0:
            df_orders['betriebszentrale_name'] = df_orders['betriebszentrale_name'].fillna('Unknown Betriebszentrale')
            print(f"\n📊 Mapping results:")
            print(f"   ✓ Mapped: {mapped_count:,} orders ({mapped_count/len(df_orders)*100:.1f}%)")
            print(f"   ⚠️  Unmapped: {unmapped_count:,} orders ({unmapped_count/len(df_orders)*100:.1f}%)")
            print(f"      → Marked as 'Unknown Betriebszentrale'")

            # Show which Auftraggeber numbers are unmapped
            unmapped_numbers = df_orders[df_orders['betriebszentrale_name'] == 'Unknown Betriebszentrale'][auftraggeber_col].unique()
            print(f"   Unmapped Auftraggeber numbers: {sorted([int(x) for x in unmapped_numbers if pd.notna(x)])}")
        else:
            print(f"\n✓ All {len(df_orders):,} orders successfully mapped to Betriebszentralen!")

        print(f"\n✓ Betriebszentralen mapping complete:")
        print(f"   Total Betriebszentralen: {df_orders['betriebszentrale_name'].nunique()}")
        print(f"   Distribution:")
        print(df_orders['betriebszentrale_name'].value_counts())

        return df_orders


class TravecomDataCleaner:
    """Data cleaning and validation utilities"""

    def __init__(self, config: Optional[ConfigLoader] = None):
        """
        Initialize data cleaner

        Args:
            config: ConfigLoader instance
        """
        self.config = config if config else ConfigLoader()

    def apply_filtering_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply business filtering rules from configuration (CORRECTED per Christian's feedback)

        Exclusions (applied in order):
        1. Lager (warehouse) orders: Lieferart 2.0 == 'Lager Auftrag'
        2. B&T pickup orders: System='B&T' AND Customer (RKdNr) is empty

        Both filters MUST be applied BEFORE any downstream analysis to ensure accurate counts.

        Args:
            df: Input DataFrame

        Returns:
            Filtered DataFrame with detailed statistics
        """
        df = df.copy()
        original_len = len(df)
        excluded_summary = []

        print(f"\n✂️  Applying Filtering Rules (Christian's Feedback - Oct 2025):")
        print(f"   Starting orders: {original_len:,}")

        # 1. FIRST: Exclude Lager (warehouse) orders
        if self.config.get('filtering.exclude_lager_orders', True):
            if 'Lieferart 2.0' in df.columns:
                before = len(df)
                mask_lager = df['Lieferart 2.0'] != 'Lager Auftrag'
                df = df[mask_lager]

                filtered_count = before - len(df)
                if filtered_count > 0:
                    excluded_summary.append(f"Lager (warehouse) orders: {filtered_count:,}")
                    print(f"   ✓ Excluded Lager orders: {filtered_count:,}")
            else:
                print(f"   ⚠️  Column 'Lieferart 2.0' not found - skipping Lager filter")

        # 2. SECOND: Exclude B&T pickup orders (System B&T with empty customer)
        # Per Christian's feedback: "System_id.Auftrag == 'B&T' AND RKdNr (customer) is empty"
        if self.config.get('filtering.exclude_bt_pickups', True):
            # Check for RKdNr (cleaned) or RKdNr. (original with dot)
            rkd_col = None
            if 'RKdNr' in df.columns:
                rkd_col = 'RKdNr'
            elif 'RKdNr.' in df.columns:
                rkd_col = 'RKdNr.'

            if 'System_id.Auftrag' in df.columns and rkd_col is not None:
                before = len(df)
                # B&T pickups: System='B&T' AND empty customer (RKdNr)
                # "Empty" includes: NaN, empty string, placeholder '-'
                # Do NOT include Auftraggeber check - Christian's rule is specific to RKdNr only
                bt_mask = (df['System_id.Auftrag'] == 'B&T')
                empty_customer_mask = df[rkd_col].isna() | df[rkd_col].isin(['-', '', ' '])

                # Exclude B&T orders with empty customer
                mask = ~(bt_mask & empty_customer_mask)
                df = df[mask]

                filtered_count = before - len(df)
                if filtered_count > 0:
                    excluded_summary.append(f"B&T pickup orders: {filtered_count:,}")
                    print(f"   ✓ Excluded B&T pickup orders: {filtered_count:,}")
                else:
                    print(f"   ℹ️  No B&T pickup orders found (already filtered or not present)")
            else:
                missing = []
                if 'System_id.Auftrag' not in df.columns:
                    missing.append('System_id.Auftrag')
                if rkd_col is None:
                    missing.append('RKdNr/RKdNr.')
                print(f"   ⚠️  Required columns not found: {missing} - skipping B&T pickup filter")

        # Print summary
        total_excluded = original_len - len(df)
        if total_excluded > 0:
            print(f"\n   📊 Filtering Summary:")
            for item in excluded_summary:
                print(f"      • {item}")
            print(f"   • Total excluded: {total_excluded:,} ({total_excluded/original_len*100:.2f}%)")
            print(f"   • Remaining orders: {len(df):,} ({len(df)/original_len*100:.2f}%)")
        else:
            print(f"\n   ✓ No orders filtered (all {len(df):,} orders passed filtering rules)")

        return df

    def validate_data(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate data and return summary statistics

        Args:
            df: Input DataFrame

        Returns:
            Dictionary with validation results
        """
        validation = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict()
        }

        print(f"\nData Validation Summary:")
        print(f"  Total rows: {validation['total_rows']:,}")
        print(f"  Total columns: {validation['total_columns']}")
        print(f"  Duplicate rows: {validation['duplicates']:,}")
        print(f"  Columns with missing values: {sum(1 for v in validation['missing_values'].values() if v > 0)}")

        return validation


def evaluate_forecast(actual: np.ndarray, predicted: np.ndarray,
                     metric: str = 'mape') -> float:
    """
    Calculate forecasting evaluation metrics

    Args:
        actual: Actual values
        predicted: Predicted values
        metric: Metric to calculate ('mape', 'rmse', 'mae')

    Returns:
        Metric value
    """
    actual = np.array(actual)
    predicted = np.array(predicted)

    if metric.lower() == 'mape':
        # Mean Absolute Percentage Error
        mask = actual != 0  # Avoid division by zero
        return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

    elif metric.lower() == 'rmse':
        # Root Mean Square Error
        return np.sqrt(np.mean((actual - predicted) ** 2))

    elif metric.lower() == 'mae':
        # Mean Absolute Error
        return np.mean(np.abs(actual - predicted))

    elif metric.lower() == 'directional_accuracy':
        # Directional Accuracy (trend prediction)
        if len(actual) < 2:
            return np.nan

        actual_direction = np.diff(actual) > 0
        predicted_direction = np.diff(predicted) > 0

        return np.mean(actual_direction == predicted_direction) * 100

    else:
        raise ValueError(f"Unknown metric: {metric}")


def calculate_multiple_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """
    Calculate multiple forecasting metrics at once

    Args:
        actual: Actual values
        predicted: Predicted values

    Returns:
        Dictionary with metric names and values
    """
    metrics = {
        'MAPE': evaluate_forecast(actual, predicted, 'mape'),
        'RMSE': evaluate_forecast(actual, predicted, 'rmse'),
        'MAE': evaluate_forecast(actual, predicted, 'mae'),
        'Directional_Accuracy': evaluate_forecast(actual, predicted, 'directional_accuracy')
    }

    return metrics


def save_processed_data(df: pd.DataFrame, filename: str, config: Optional[ConfigLoader] = None):
    """
    Save processed data to CSV

    Args:
        df: DataFrame to save
        filename: Output filename
        config: ConfigLoader instance
    """
    if config is None:
        config = ConfigLoader()

    output_path = Path(config.get('data.processed_path')) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Saved {len(df):,} rows to: {output_path}")


def load_processed_data(filename: str, config: Optional[ConfigLoader] = None) -> pd.DataFrame:
    """
    Load processed data from CSV

    Args:
        filename: Input filename
        config: ConfigLoader instance

    Returns:
        Loaded DataFrame
    """
    if config is None:
        config = ConfigLoader()

    input_path = Path(config.get('data.processed_path')) / filename

    if not input_path.exists():
        raise FileNotFoundError(f"Processed data file not found: {input_path}")

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} rows from: {input_path}")

    return df


# Example usage
if __name__ == "__main__":
    print("Traveco Forecasting Utilities")
    print("=" * 50)

    # Test configuration loading
    config = ConfigLoader()
    print(f"\nTarget columns: {config.get('features.target_columns')}")
    print(f"Lag periods: {config.get('features.lag_periods')}")

    # Test data loading
    loader = TravecomDataLoader(config)
    print(f"\nData path: {loader.data_path}")

    print("\nUtilities ready!")
