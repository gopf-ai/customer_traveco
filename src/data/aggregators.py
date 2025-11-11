"""Data aggregation for Traveco forecasting system"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from src.utils.logging_config import get_logger


logger = get_logger(__name__)


class DataAggregator:
    """
    Aggregate order and tour data to monthly level

    Handles:
    - Betriebszentralen-level aggregation
    - Company-level aggregation
    - Tour data merging
    - Working days integration
    - Personnel costs integration
    - Total revenue integration
    """

    def __init__(self, config: Dict = None):
        """
        Initialize aggregator

        Args:
            config: Configuration dictionary
        """
        self.config = config if config else {}

    def aggregate_orders_monthly(
        self,
        df_orders: pd.DataFrame,
        group_by: List[str] = None
    ) -> pd.DataFrame:
        """
        Aggregate orders to monthly level

        Args:
            df_orders: DataFrame with order data
            group_by: Columns to group by (default: ['year_month', 'betriebszentrale_name'])

        Returns:
            DataFrame aggregated to monthly level
        """
        if group_by is None:
            group_by = ['year_month']
            if 'betriebszentrale_name' in df_orders.columns:
                group_by.append('betriebszentrale_name')

        logger.info(f"Aggregating {len(df_orders):,} orders by {group_by}")

        # Ensure year_month exists
        if 'year_month' not in df_orders.columns:
            # Try to find and use a date column
            date_col = None

            # Check for possible date columns (in order of preference)
            possible_date_cols = ['date', 'Datum.Tour', 'Datum.Auftrag']

            for col in possible_date_cols:
                if col in df_orders.columns:
                    date_col = col
                    break

            if date_col is None:
                raise ValueError(f"No date column found. Tried: {possible_date_cols}. Available columns: {df_orders.columns.tolist()[:10]}...")

            # Create standardized date column if needed
            if date_col != 'date':
                df_orders['date'] = pd.to_datetime(df_orders[date_col])
                logger.info(f"  Created 'date' column from '{date_col}'")

            # Create year_month
            df_orders['year_month'] = df_orders['date'].dt.to_period('M').astype(str)
            logger.info(f"  Created 'year_month' column")

        # Define aggregation rules
        agg_dict = {
            'Nummer.Auftrag': 'count',  # total_orders
        }

        # Revenue (Column AV: ∑ Einnahmen)
        if '∑ Einnahmen' in df_orders.columns:
            agg_dict['∑ Einnahmen'] = 'sum'

        # Distance (Column CU: Distanz_BE.Auftrag)
        if 'Distanz_BE.Auftrag' in df_orders.columns:
            agg_dict['Distanz_BE.Auftrag'] = 'sum'

        # Carrier counts
        if 'carrier_type' in df_orders.columns:
            # We'll handle this separately for internal/external split
            pass

        # Perform aggregation
        df_monthly = df_orders.groupby(group_by).agg(agg_dict).reset_index()

        # Rename columns
        column_mapping = {
            'Nummer.Auftrag': 'total_orders',
            '∑ Einnahmen': 'revenue_total',
            'Distanz_BE.Auftrag': 'total_km_billed'
        }
        df_monthly = df_monthly.rename(columns=column_mapping)

        # Add driver counts (internal vs external)
        if 'carrier_type' in df_orders.columns:
            driver_counts = df_orders.groupby(group_by + ['carrier_type']).size().unstack(fill_value=0)

            if 'internal' in driver_counts.columns:
                df_monthly = df_monthly.merge(
                    driver_counts[['internal']].rename(columns={'internal': 'internal_drivers'}),
                    on=group_by,
                    how='left'
                )

            if 'external' in driver_counts.columns:
                df_monthly = df_monthly.merge(
                    driver_counts[['external']].rename(columns={'external': 'external_drivers'}),
                    on=group_by,
                    how='left'
                )

        # Calculate total drivers
        if 'internal_drivers' in df_monthly.columns and 'external_drivers' in df_monthly.columns:
            df_monthly['total_drivers'] = (
                df_monthly['internal_drivers'].fillna(0) +
                df_monthly['external_drivers'].fillna(0)
            )

        # Add date column
        if 'date' not in df_monthly.columns:
            df_monthly['date'] = pd.to_datetime(df_monthly['year_month'] + '-01')

        logger.info(f"Aggregated to {len(df_monthly):,} monthly records")

        return df_monthly

    def merge_tour_data(
        self,
        df_monthly: pd.DataFrame,
        df_tours: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge tour-level metrics into monthly aggregation

        Tour metrics include:
        - total_tours (count of unique tours)
        - total_km_actual (sum of Soll KM PraCar)
        - vehicle_km_cost (KM × KM Cost)
        - vehicle_time_cost (Time × Time Cost)
        - total_vehicle_cost (sum of both)

        Args:
            df_monthly: Monthly aggregated order data
            df_tours: Tour data

        Returns:
            DataFrame with tour metrics merged
        """
        logger.info(f"Merging tour data ({len(df_tours):,} tours)")

        # Ensure year_month exists in tours
        if 'year_month' not in df_tours.columns and 'date' in df_tours.columns:
            df_tours['year_month'] = df_tours['date'].dt.to_period('M').astype(str)

        # Calculate tour-level costs
        # First, ensure all cost columns are numeric (they may contain strings from Excel)
        cost_columns_to_convert = ['IstKm.Tour', 'Soll KM PraCar', 'PC KM Kosten',
                                    'IST Zeit PraCar', 'PC Minuten Kosten']
        for col in cost_columns_to_convert:
            if col in df_tours.columns:
                df_tours[col] = pd.to_numeric(df_tours[col], errors='coerce')

        # Now perform calculations
        if 'IstKm.Tour' in df_tours.columns and 'PC KM Kosten' in df_tours.columns:
            df_tours['vehicle_km_cost'] = df_tours['IstKm.Tour'].fillna(0) * df_tours['PC KM Kosten'].fillna(0)
        elif 'Soll KM PraCar' in df_tours.columns and 'PC KM Kosten' in df_tours.columns:
            df_tours['vehicle_km_cost'] = df_tours['Soll KM PraCar'].fillna(0) * df_tours['PC KM Kosten'].fillna(0)

        if 'IST Zeit PraCar' in df_tours.columns and 'PC Minuten Kosten' in df_tours.columns:
            df_tours['vehicle_time_cost'] = (
                df_tours['IST Zeit PraCar'].fillna(0) * 60 * df_tours['PC Minuten Kosten'].fillna(0)
            )

        if 'vehicle_km_cost' in df_tours.columns and 'vehicle_time_cost' in df_tours.columns:
            df_tours['total_vehicle_cost'] = df_tours['vehicle_km_cost'] + df_tours['vehicle_time_cost']

        # Determine grouping columns
        group_cols = ['year_month']
        if 'betriebszentrale_name' in df_tours.columns and 'betriebszentrale_name' in df_monthly.columns:
            group_cols.append('betriebszentrale_name')

        # Aggregate tours
        tour_agg_dict = {
            'Nummer.Tour': 'nunique',  # Count unique tours
        }

        if 'Soll KM PraCar' in df_tours.columns:
            tour_agg_dict['Soll KM PraCar'] = 'sum'

        if 'vehicle_km_cost' in df_tours.columns:
            tour_agg_dict['vehicle_km_cost'] = 'sum'

        if 'vehicle_time_cost' in df_tours.columns:
            tour_agg_dict['vehicle_time_cost'] = 'sum'

        if 'total_vehicle_cost' in df_tours.columns:
            tour_agg_dict['total_vehicle_cost'] = 'sum'

        df_tour_monthly = df_tours.groupby(group_cols).agg(tour_agg_dict).reset_index()

        # Rename columns
        tour_column_mapping = {
            'Nummer.Tour': 'total_tours',
            'Soll KM PraCar': 'total_km_actual'
        }
        df_tour_monthly = df_tour_monthly.rename(columns=tour_column_mapping)

        # Merge with monthly data
        df_merged = df_monthly.merge(df_tour_monthly, on=group_cols, how='left')

        logger.info(f"Tour data merged: {len(df_merged):,} records")

        return df_merged

    def add_working_days(
        self,
        df_monthly: pd.DataFrame,
        df_working_days: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add working days to monthly data

        Args:
            df_monthly: Monthly aggregated data
            df_working_days: Working days data (year, month, working_days)

        Returns:
            DataFrame with working_days column added
        """
        if df_working_days.empty:
            logger.warning("Working days data is empty, skipping integration")
            return df_monthly

        logger.info("Adding working days data")

        # Ensure year and month columns exist
        if 'year' not in df_monthly.columns:
            df_monthly['year'] = df_monthly['date'].dt.year
        if 'month' not in df_monthly.columns:
            df_monthly['month'] = df_monthly['date'].dt.month

        # Merge on year and month
        df_merged = df_monthly.merge(
            df_working_days[['year', 'month', 'working_days']],
            on=['year', 'month'],
            how='left'
        )

        # Log statistics
        missing_count = df_merged['working_days'].isna().sum()
        if missing_count > 0:
            logger.warning(f"Working days missing for {missing_count} months")
        else:
            logger.info(f"✅ Working days added for all {len(df_merged)} months")

        return df_merged

    def add_personnel_costs(
        self,
        df_monthly: pd.DataFrame,
        df_personnel: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add personnel costs to monthly data (NEW)

        Args:
            df_monthly: Monthly aggregated data
            df_personnel: Personnel costs data (date, personnel_costs)

        Returns:
            DataFrame with personnel_costs column added
        """
        if df_personnel.empty:
            logger.warning("Personnel costs data is empty, skipping integration")
            return df_monthly

        logger.info("Adding personnel costs data")

        # Ensure date column is datetime
        df_personnel['date'] = pd.to_datetime(df_personnel['date'])

        # Create year_month for matching
        df_personnel['year_month'] = df_personnel['date'].dt.to_period('M').astype(str)

        # Merge on year_month
        df_merged = df_monthly.merge(
            df_personnel[['year_month', 'personnel_costs']],
            on='year_month',
            how='left'
        )

        # Log statistics
        missing_count = df_merged['personnel_costs'].isna().sum()
        if missing_count > 0:
            logger.warning(f"Personnel costs missing for {missing_count} months")
        else:
            logger.info(f"✅ Personnel costs added for all {len(df_merged)} months")

        return df_merged

    def add_total_revenue(
        self,
        df_monthly: pd.DataFrame,
        df_total_revenue: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add total revenue (transportation + other) to monthly data (NEW)

        Args:
            df_monthly: Monthly aggregated data
            df_total_revenue: Total revenue data (date, total_revenue_all)

        Returns:
            DataFrame with total_revenue_all column added
        """
        if df_total_revenue.empty:
            logger.warning("Total revenue data is empty, skipping integration")
            return df_monthly

        logger.info("Adding total revenue data")

        # Ensure date column is datetime
        df_total_revenue['date'] = pd.to_datetime(df_total_revenue['date'])

        # Create year_month for matching
        df_total_revenue['year_month'] = df_total_revenue['date'].dt.to_period('M').astype(str)

        # Merge on year_month
        df_merged = df_monthly.merge(
            df_total_revenue[['year_month', 'total_revenue_all']],
            on='year_month',
            how='left'
        )

        # Validate: total_revenue_all should be >= revenue_total
        if 'revenue_total' in df_merged.columns and 'total_revenue_all' in df_merged.columns:
            violations = df_merged[df_merged['total_revenue_all'] < df_merged['revenue_total']]

            if len(violations) > 0:
                logger.error(
                    f"❌ Found {len(violations)} months where total_revenue_all < revenue_total!"
                )
                for idx, row in violations.head(3).iterrows():
                    logger.error(
                        f"  {row['year_month']}: transport={row['revenue_total']:,.0f}, "
                        f"total={row['total_revenue_all']:,.0f}"
                    )
            else:
                logger.info("✅ Revenue relationship validation passed (total ≥ transport)")

        # Log statistics
        missing_count = df_merged['total_revenue_all'].isna().sum()
        if missing_count > 0:
            logger.warning(f"Total revenue missing for {missing_count} months")
        else:
            logger.info(f"✅ Total revenue added for all {len(df_merged)} months")

        return df_merged

    def aggregate_to_company_level(
        self,
        df_bz_level: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Aggregate from Betriebszentralen-level to company-level

        Args:
            df_bz_level: DataFrame at Betriebszentralen level

        Returns:
            DataFrame aggregated to company level (monthly)
        """
        logger.info("Aggregating to company level")

        if 'betriebszentrale_name' not in df_bz_level.columns:
            logger.warning("Data already at company level, skipping aggregation")
            return df_bz_level

        # Define aggregation rules
        agg_dict = {
            'total_orders': 'sum',
            'revenue_total': 'sum',
            'total_km_billed': 'sum',
            'internal_drivers': 'sum',
            'external_drivers': 'sum',
            'total_drivers': 'sum',
        }

        # Add tour metrics if present
        tour_metrics = ['total_tours', 'total_km_actual', 'vehicle_km_cost',
                       'vehicle_time_cost', 'total_vehicle_cost']
        for metric in tour_metrics:
            if metric in df_bz_level.columns:
                agg_dict[metric] = 'sum'

        # Working days: use 'first' (same for all BZ in a month)
        if 'working_days' in df_bz_level.columns:
            agg_dict['working_days'] = 'first'

        # Personnel costs: use 'first' (company-level metric)
        if 'personnel_costs' in df_bz_level.columns:
            agg_dict['personnel_costs'] = 'first'

        # Total revenue: use 'first' (company-level metric)
        if 'total_revenue_all' in df_bz_level.columns:
            agg_dict['total_revenue_all'] = 'first'

        # Aggregate by year_month
        df_company = df_bz_level.groupby('year_month').agg(agg_dict).reset_index()

        # Add date column
        if 'date' not in df_company.columns:
            df_company['date'] = pd.to_datetime(df_company['year_month'] + '-01')

        # Add month and year
        df_company['month'] = df_company['date'].dt.month
        df_company['year'] = df_company['date'].dt.year

        # Calculate derived metrics
        if 'revenue_total' in df_company.columns and 'total_orders' in df_company.columns:
            df_company['revenue_per_order'] = df_company['revenue_total'] / df_company['total_orders']

        if 'total_km_billed' in df_company.columns and 'total_orders' in df_company.columns:
            df_company['km_per_order'] = df_company['total_km_billed'] / df_company['total_orders']

        if 'total_km_actual' in df_company.columns and 'total_km_billed' in df_company.columns:
            df_company['km_efficiency'] = df_company['total_km_actual'] / df_company['total_km_billed']

        logger.info(f"Aggregated to {len(df_company):,} company-level monthly records")

        return df_company

    def create_full_time_series(
        self,
        df_orders: pd.DataFrame,
        df_tours: pd.DataFrame,
        df_working_days: pd.DataFrame = None,
        df_personnel: pd.DataFrame = None,
        df_total_revenue: pd.DataFrame = None,
        company_level: bool = True
    ) -> pd.DataFrame:
        """
        Create complete monthly time series with all metrics

        This is the main orchestration method that combines all data sources.

        Args:
            df_orders: Order data
            df_tours: Tour data
            df_working_days: Working days data (optional)
            df_personnel: Personnel costs data (optional)
            df_total_revenue: Total revenue data (optional)
            company_level: If True, aggregate to company level

        Returns:
            Complete monthly time series DataFrame
        """
        logger.info("=" * 60)
        logger.info("Creating full time series")
        logger.info("=" * 60)

        # Step 1: Aggregate orders
        df_monthly = self.aggregate_orders_monthly(df_orders)

        # Step 2: Merge tour data
        if not df_tours.empty:
            df_monthly = self.merge_tour_data(df_monthly, df_tours)

        # Step 3: Add working days
        if df_working_days is not None and not df_working_days.empty:
            df_monthly = self.add_working_days(df_monthly, df_working_days)

        # Step 4: Aggregate to company level (if requested)
        if company_level:
            df_monthly = self.aggregate_to_company_level(df_monthly)

        # Step 5: Add personnel costs (company-level only)
        if df_personnel is not None and not df_personnel.empty:
            df_monthly = self.add_personnel_costs(df_monthly, df_personnel)

        # Step 6: Add total revenue (company-level only)
        if df_total_revenue is not None and not df_total_revenue.empty:
            df_monthly = self.add_total_revenue(df_monthly, df_total_revenue)

        # Sort by date
        df_monthly = df_monthly.sort_values('date').reset_index(drop=True)

        logger.info("=" * 60)
        logger.info(f"Time series complete: {len(df_monthly)} months")
        logger.info(f"Date range: {df_monthly['date'].min()} to {df_monthly['date'].max()}")
        logger.info(f"Columns: {len(df_monthly.columns)}")
        logger.info("=" * 60)

        return df_monthly
