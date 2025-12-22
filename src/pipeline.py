"""Core forecasting pipeline orchestration

This module implements the main ForecastingPipeline class that orchestrates
the entire forecasting workflow from data loading to model training to
forecast generation and validation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle
from datetime import datetime
import json

from src.data import DataLoader, DataCleaner, DataValidator, DataAggregator
from src.features import FeatureEngine
from src.models.xgboost_forecaster import XGBoostForecaster
from src.models.baseline_forecasters import (
    SeasonalNaiveForecaster,
    MovingAverageForecaster,
    LinearTrendForecaster
)
from src.revenue.percentage_model import RevenuePercentageModel
from src.revenue.ml_model import RevenueMLModel
from src.revenue.ensemble import RevenueEnsemble
from src.utils.config import ConfigLoader
from src.utils.logging_config import get_logger

# Rich components for progress tracking
try:
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.console import Console
    RICH_AVAILABLE = True
    rich_console = Console()
except ImportError:
    RICH_AVAILABLE = False
    rich_console = None


logger = get_logger(__name__)


class ForecastingPipeline:
    """
    Main forecasting pipeline orchestrator

    Coordinates the entire forecasting workflow:
    1. Data loading and validation
    2. Feature engineering
    3. Model training (forecasting + revenue)
    4. Forecast generation
    5. Validation and reporting
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Forecasting Pipeline

        Args:
            config_path: Path to config YAML file (optional)
        """
        self.config = ConfigLoader(config_path)

        # Initialize components
        self.loader = DataLoader(self.config)
        self.cleaner = DataCleaner(self.config)
        self.validator = DataValidator(self.config)
        self.aggregator = DataAggregator(self.config)
        self.feature_engine = FeatureEngine(self.config)

        # Models (initialized after training)
        self.forecasting_models: Dict[str, Dict] = {}
        self.revenue_ensemble: Optional[RevenueEnsemble] = None

        # Data cache
        self.df_historic: Optional[pd.DataFrame] = None
        self.df_time_series: Optional[pd.DataFrame] = None

        # Paths
        self.models_dir = Path(self.config.get('paths.models_dir', 'models'))
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.forecasts_dir = Path(self.config.get('paths.forecasts_dir', 'forecasts'))
        self.forecasts_dir.mkdir(parents=True, exist_ok=True)

        self.reports_dir = Path(self.config.get('paths.reports_dir', 'reports'))
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def _validate_training_data(self, df: pd.DataFrame) -> None:
        """
        Validate data quality before training

        Checks:
        1. Dates are valid (not epoch/1970)
        2. Minimum 36 months of data
        3. Sufficient historical depth

        Args:
            df: DataFrame with training data

        Raises:
            ValueError: If data fails validation checks
        """
        # Check 1: Validate dates exist and are valid
        if 'date' not in df.columns:
            raise ValueError("Training data must have a 'date' column")

        df['date'] = pd.to_datetime(df['date'])
        min_date = df['date'].min()
        max_date = df['date'].max()

        # Check for invalid dates (Unix epoch or before year 2000)
        if pd.isna(min_date) or pd.isna(max_date):
            raise ValueError(
                "Invalid dates detected: NaT (Not a Time) values found. "
                "Check date column formatting in source data."
            )

        if min_date.year < 2000:
            raise ValueError(
                f"Invalid date range detected: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}. "
                f"Dates before year 2000 indicate data loading issues. "
                f"Check Excel date format and column mapping."
            )

        # Check 2: Require minimum 36 months (3 years)
        months_available = len(df)
        if months_available < 36:
            raise ValueError(
                f"Insufficient data for training: {months_available} months found. "
                f"Minimum required: 36 months (3 years). "
                f"Current date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}. "
                f"Please provide complete historical data (2022-2024) or use --use-processed."
            )

        # Check 3: Warn if data span is too short (< 2.5 years)
        years_span = (max_date - min_date).days / 365.25
        if years_span < 2.5:
            logger.warning(
                f"⚠️  Limited historical data: {years_span:.1f} years span. "
                f"Forecasts may be less reliable with < 3 years of history."
            )

        logger.info(f"✅ Data validation passed: {months_available} months, {years_span:.1f} years span")

    def build_master_file(
        self,
        years: List[int] = None,
        output_formats: List[str] = None,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Build master historical dataset from raw Excel files

        Loads all monthly files across multiple years, applies feature engineering,
        filtering rules, and saves to processed formats.

        Args:
            years: List of years to process (default: [2022, 2023, 2024, 2025])
            output_formats: List of formats to save: pkl, csv, parquet (default: [pkl, csv])
            output_path: Custom output path (default: data/processed/historic_orders_YYYY_YYYY.pkl)

        Returns:
            DataFrame with complete historical data
        """
        if years is None:
            years = self.config.get('data.historic_years', [2022, 2023, 2024, 2025])

        if output_formats is None:
            output_formats = ['pkl', 'csv']

        logger.info(f"Building master file from {len(years)} years: {years}")
        logger.info(f"Output formats: {output_formats}")

        # Stage 1: Load raw data
        logger.info("\n[1/5] Loading raw Excel files...")
        df_orders = self.loader.load_historic_orders_multi_year(years=years)
        df_tours = self.loader.load_historic_tours_multi_year(years=years)

        logger.info(f"✓ Loaded {len(df_orders):,} orders")
        if not df_tours.empty:
            logger.info(f"✓ Loaded {len(df_tours):,} tours")

        # Stage 2: Apply data cleaning
        logger.info("\n[2/5] Applying data cleaning and validation...")
        # Note: Cleaner and validator would be applied here if needed
        # For now, we assume the aggregator handles basic cleaning

        # Stage 3: Load reference data
        logger.info("\n[3/5] Loading reference data...")
        df_divisions = self.loader.load_divisions()
        df_betriebszentralen = self.loader.load_betriebszentralen()

        # Stage 4: Apply feature engineering
        logger.info("\n[4/5] Applying feature engineering...")

        # Import utilities from utils (has all the methods from notebooks)
        from utils.traveco_utils import TravecomFeatureEngine, TravecomDataCleaner
        feature_engine = TravecomFeatureEngine(self.config)
        data_cleaner = TravecomDataCleaner(self.config)

        # Apply temporal features
        logger.info("  - Extracting temporal features...")
        # First, detect and convert date column
        date_col = None
        for col in ['date', 'Datum.Tour', 'Datum.Auftrag']:
            if col in df_orders.columns:
                date_col = col
                break

        if date_col and date_col != 'date':
            df_orders['date'] = feature_engine.convert_date_column(df_orders[date_col])

        df_orders = feature_engine.extract_temporal_features(df_orders, date_column='date')

        # Apply carrier type classification
        logger.info("  - Identifying carrier types...")
        if 'Nummer.Spedition' in df_orders.columns:
            df_orders = feature_engine.identify_carrier_type(df_orders)

        # Apply Betriebszentralen mapping
        logger.info("  - Mapping Betriebszentralen...")
        if not df_betriebszentralen.empty:
            df_orders = feature_engine.map_betriebszentralen(df_orders, df_betriebszentralen)

        # Apply Sparten mapping
        logger.info("  - Mapping customer divisions (Sparten)...")
        if not df_divisions.empty:
            df_orders = feature_engine.map_customer_divisions(df_orders, df_divisions)

        # Apply filtering rules
        logger.info("  - Applying filtering rules...")
        df_orders = data_cleaner.apply_filtering_rules(df_orders)

        # Classify order types
        logger.info("  - Classifying order types...")
        df_orders = feature_engine.classify_order_type_multifield(df_orders)

        logger.info(f"✓ Feature engineering complete: {len(df_orders.columns)} columns")

        # Stage 5: Save to files
        logger.info("\n[5/5] Saving master file...")

        if output_path is None:
            year_min = min(years)
            year_max = max(years)
            output_base = Path(self.config.get('paths.processed_data_dir', 'data/processed'))
            output_base.mkdir(parents=True, exist_ok=True)
            output_name = f"historic_orders_{year_min}_{year_max}"
        else:
            output_base = Path(output_path).parent
            output_name = Path(output_path).stem

        saved_files = []

        # Save pickle (fastest for loading)
        if 'pkl' in output_formats:
            pkl_path = output_base / f"{output_name}.pkl"
            logger.info(f"  Saving pickle: {pkl_path}")
            df_orders.to_pickle(pkl_path)
            saved_files.append((pkl_path, pkl_path.stat().st_size / (1024**3)))  # Size in GB

        # Save compressed CSV (portable)
        if 'csv' in output_formats:
            csv_path = output_base / f"{output_name}.csv.gz"
            logger.info(f"  Saving compressed CSV: {csv_path}")
            df_orders.to_csv(csv_path, index=False, compression='gzip')
            saved_files.append((csv_path, csv_path.stat().st_size / (1024**3)))

        # Save parquet (optional, good balance)
        if 'parquet' in output_formats:
            parquet_path = output_base / f"{output_name}.parquet"
            logger.info(f"  Saving parquet: {parquet_path}")
            df_orders.to_parquet(parquet_path, index=False)
            saved_files.append((parquet_path, parquet_path.stat().st_size / (1024**3)))

        # Summary
        logger.info("\n" + "="*60)
        logger.info("MASTER FILE BUILD COMPLETE")
        logger.info("="*60)
        logger.info(f"Records:  {len(df_orders):,}")
        logger.info(f"Columns:  {len(df_orders.columns)}")
        logger.info(f"Date Range: {df_orders['date'].min()} to {df_orders['date'].max()}")
        logger.info(f"Years:    {sorted(df_orders['_source_year'].unique().tolist())}")
        logger.info("\nSaved Files:")
        for file_path, size_gb in saved_files:
            logger.info(f"  {file_path.name:40s} {size_gb:>6.2f} GB")
        logger.info("="*60 + "\n")

        return df_orders

    def load_processed_data(
        self,
        file_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load pre-processed historical time series data

        Args:
            file_path: Path to processed CSV file (default: from config)

        Returns:
            DataFrame with monthly time series
        """
        if file_path is None:
            processed_dir = Path(self.config.get('paths.processed_data_dir', 'data/processed'))
            file_path = processed_dir / 'monthly_aggregated_full_company.csv'
        else:
            file_path = Path(file_path)

        logger.info(f"Loading processed data from: {file_path}")

        if not file_path.exists():
            raise FileNotFoundError(f"Processed data file not found: {file_path}")

        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])

        logger.info(f"✅ Loaded {len(df)} months of processed data")
        logger.info(f"   Date range: {df['date'].min()} to {df['date'].max()}")

        return df

    def load_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        validate: bool = True,
        use_processed: bool = False,
        data_source: str = 'processed',
        custom_file: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load and prepare historical data

        Args:
            start_date: Start date for filtering (YYYY-MM-DD format)
            end_date: End date for filtering (YYYY-MM-DD format)
            validate: Whether to run data validation
            use_processed: [DEPRECATED] Use data_source='processed' instead
            data_source: Data source type: 'processed', 'historic', 'validation', 'custom'
            custom_file: Custom file path (required if data_source='custom')

        Returns:
            DataFrame with complete time series
        """
        logger.info("=" * 80)
        logger.info("LOADING HISTORICAL DATA")
        logger.info("=" * 80)

        # Handle deprecated use_processed parameter
        if use_processed and data_source == 'processed':
            data_source = 'processed'
        elif not use_processed and data_source == 'processed':
            data_source = 'validation'  # Legacy behavior: load raw = validation

        # Option 1: Load from processed aggregated data (fastest)
        if data_source == 'processed':
            logger.info("Loading from processed data...")
            df_time_series = self.load_processed_data()

            # Filter by date range if specified
            if start_date or end_date:
                df_time_series['date'] = pd.to_datetime(df_time_series['date'])

                if start_date:
                    df_time_series = df_time_series[df_time_series['date'] >= start_date]
                    logger.info(f"Filtered to start_date >= {start_date}")

                if end_date:
                    df_time_series = df_time_series[df_time_series['date'] <= end_date]
                    logger.info(f"Filtered to end_date <= {end_date}")

            # Validate data before returning
            if validate:
                self._validate_training_data(df_time_series)

            self.df_time_series = df_time_series
            return df_time_series

        # Option 2: Load from historic master file (pkl)
        elif data_source == 'historic':
            logger.info("Loading from historic master file...")

            # Try to load from pickle file
            processed_dir = Path(self.config.get('paths.processed_data_dir', 'data/processed'))
            historic_files = list(processed_dir.glob('historic_orders_*_*.pkl'))

            if not historic_files:
                raise FileNotFoundError(
                    f"No historic master file found in {processed_dir}. "
                    f"Run 'traveco build-master' first to create it."
                )

            # Use most recent file (sorted by name, which includes years)
            historic_file = sorted(historic_files)[-1]
            logger.info(f"Loading: {historic_file}")

            df_orders = pd.read_pickle(historic_file)
            logger.info(f"✓ Loaded {len(df_orders):,} orders from historic master file")

            # Aggregate to monthly time series
            logger.info("Aggregating to monthly time series...")
            df_time_series = self.aggregator.create_full_time_series(
                df_orders=df_orders,
                df_tours=pd.DataFrame(),  # Tours already merged if needed
                df_working_days=None,
                df_personnel=None,
                df_total_revenue=None
            )

            # Filter by date range if specified
            if start_date or end_date:
                df_time_series['date'] = pd.to_datetime(df_time_series['date'])

                if start_date:
                    df_time_series = df_time_series[df_time_series['date'] >= start_date]
                    logger.info(f"Filtered to start_date >= {start_date}")

                if end_date:
                    df_time_series = df_time_series[df_time_series['date'] <= end_date]
                    logger.info(f"Filtered to end_date <= {end_date}")

            # Validate data before returning
            if validate:
                self._validate_training_data(df_time_series)

            self.df_time_series = df_time_series
            return df_time_series

        # Option 3: Load from custom file
        elif data_source == 'custom':
            if custom_file is None:
                raise ValueError("custom_file path required when data_source='custom'")

            logger.info(f"Loading from custom file: {custom_file}")
            custom_path = Path(custom_file)

            if not custom_path.exists():
                raise FileNotFoundError(f"Custom file not found: {custom_file}")

            # Detect file type and load accordingly
            if custom_path.suffix == '.pkl':
                df = pd.read_pickle(custom_path)
            elif custom_path.suffix == '.parquet':
                df = pd.read_parquet(custom_path)
            elif custom_path.suffix in ['.csv', '.gz']:
                df = pd.read_csv(custom_path)
                df['date'] = pd.to_datetime(df['date'])
            else:
                raise ValueError(f"Unsupported file type: {custom_path.suffix}")

            logger.info(f"✓ Loaded {len(df):,} records from custom file")

            # Validate data before returning
            if validate:
                self._validate_training_data(df)

            self.df_time_series = df
            return df

        # Option 4: Load from validation data (single month raw file)
        elif data_source == 'validation':
            logger.info("Loading from validation data (raw single-month file)...")
        # Option 5: Legacy fallback - raw data sources
        else:
            logger.info("Loading from raw data sources...")

        # Load all data sources
        logger.info("Loading orders data...")
        df_orders = self.loader.load_orders()

        logger.info("Loading tours data...")
        df_tours = self.loader.load_tours()

        logger.info("Loading working days data...")
        df_working_days = self.loader.load_working_days()

        # Load NEW data sources if available
        df_personnel = self.loader.load_personnel_costs()
        df_total_revenue = self.loader.load_total_revenue()

        # Log optional data availability summary
        optional_data_status = []
        if df_personnel is not None and not df_personnel.empty:
            optional_data_status.append("personnel_costs ✓")
        else:
            optional_data_status.append("personnel_costs ✗")
            df_personnel = None

        if df_total_revenue is not None and not df_total_revenue.empty:
            optional_data_status.append("total_revenue ✓")
        else:
            optional_data_status.append("total_revenue ✗")
            df_total_revenue = None

        logger.info(f"Optional data sources: {' | '.join(optional_data_status)}")

        # Clean data
        logger.info("Cleaning orders data...")
        df_orders_clean = self.cleaner.clean_orders(df_orders)

        logger.info("Cleaning tours data...")
        df_tours_clean = self.cleaner.clean_tours(df_tours)

        # Validate if requested
        if validate:
            logger.info("Validating data quality...")

            validation_report = self.validator.validate_dataframe(
                df_orders_clean,
                'orders'
            )

            logger.info("Validation Report:")
            logger.info(f"  Total Records: {validation_report['total_records']}")
            logger.info(f"  Complete Records: {validation_report['complete_records']}")
            logger.info(f"  Missing Values: {validation_report['missing_values']}")
            logger.info(f"  Outliers Detected: {validation_report['outliers_detected']}")

        # Create time series
        logger.info("Creating monthly time series...")
        df_time_series = self.aggregator.create_full_time_series(
            df_orders=df_orders_clean,
            df_tours=df_tours_clean,
            df_working_days=df_working_days,
            df_personnel=df_personnel,
            df_total_revenue=df_total_revenue,
            company_level=True
        )

        # Filter by date range if specified
        if start_date or end_date:
            df_time_series['date'] = pd.to_datetime(df_time_series['date'])

            if start_date:
                df_time_series = df_time_series[df_time_series['date'] >= start_date]
                logger.info(f"Filtered to start_date >= {start_date}")

            if end_date:
                df_time_series = df_time_series[df_time_series['date'] <= end_date]
                logger.info(f"Filtered to end_date <= {end_date}")

        # Validate data quality
        self._validate_training_data(df_time_series)

        # Cache data
        self.df_time_series = df_time_series

        logger.info(f"✅ Data loaded successfully: {len(df_time_series)} months")
        logger.info(f"   Date range: {df_time_series['date'].min()} to {df_time_series['date'].max()}")

        return df_time_series

    def train_models(
        self,
        df_train: Optional[pd.DataFrame] = None,
        train_baseline: bool = True,
        train_xgboost: bool = True,
        train_revenue: bool = True,
        save_models: bool = True,
        xgboost_validation_months: int = 6,
        xgboost_cv_folds: Optional[int] = None
    ) -> Dict[str, Dict]:
        """
        Train all forecasting models

        Args:
            df_train: Training data (uses cached data if None)
            train_baseline: Train baseline models
            train_xgboost: Train XGBoost models
            train_revenue: Train revenue models
            save_models: Save trained models to disk
            xgboost_validation_months: Number of months for XGBoost validation holdout
            xgboost_cv_folds: If specified, use time series CV with N folds for XGBoost

        Returns:
            Dictionary of trained models with metadata
        """
        logger.info("=" * 80)
        logger.info("TRAINING FORECASTING MODELS")
        logger.info("=" * 80)

        # Use cached data if not provided
        if df_train is None:
            if self.df_time_series is None:
                raise ValueError("No training data available. Run load_data() first.")
            df_train = self.df_time_series.copy()

        # Get core metrics to forecast
        core_metrics = self.config.get('features.core_metrics', [
            'revenue_total',
            'personnel_costs',
            'external_drivers'
        ])

        # Filter to available metrics
        available_metrics = [m for m in core_metrics if m in df_train.columns]
        skipped_metrics = [m for m in core_metrics if m not in df_train.columns]

        if skipped_metrics:
            logger.info(f"⚠️  Skipping unavailable metrics: {', '.join(skipped_metrics)}")

        logger.info(f"Training models for {len(available_metrics)} metrics: {available_metrics}")

        # Calculate total models to train
        models_per_metric = 0
        if train_baseline:
            models_per_metric += 3  # Seasonal Naive, MA, Linear Trend
        if train_xgboost:
            models_per_metric += 1  # XGBoost
        total_models = len(available_metrics) * models_per_metric

        # Use rich Progress if available, otherwise fallback to regular logging
        if RICH_AVAILABLE and rich_console:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("({task.completed}/{task.total})"),
                TimeElapsedColumn(),
                console=rich_console
            ) as progress:
                overall_task = progress.add_task(
                    "[cyan]Training models...",
                    total=total_models
                )

                # Train models for each metric
                for metric_idx, metric in enumerate(available_metrics, 1):
                    metric_task = progress.add_task(
                        f"  [yellow]{metric}",
                        total=models_per_metric
                    )

                    self.forecasting_models[metric] = {}

                    # Baseline models
                    if train_baseline:
                        # Seasonal Naive
                        progress.update(metric_task, description=f"  [yellow]{metric}[/yellow] → Seasonal Naive")
                        sn_model = SeasonalNaiveForecaster()
                        sn_model.fit(df_train, metric)
                        self.forecasting_models[metric]['seasonal_naive'] = sn_model
                        progress.update(metric_task, advance=1)
                        progress.update(overall_task, advance=1)

                        # Moving Average
                        progress.update(metric_task, description=f"  [yellow]{metric}[/yellow] → Moving Average")
                        ma_model = MovingAverageForecaster(window=3)
                        ma_model.fit(df_train, metric)
                        self.forecasting_models[metric]['moving_average'] = ma_model
                        progress.update(metric_task, advance=1)
                        progress.update(overall_task, advance=1)

                        # Linear Trend
                        progress.update(metric_task, description=f"  [yellow]{metric}[/yellow] → Linear Trend")
                        lt_model = LinearTrendForecaster()
                        lt_model.fit(df_train, metric)
                        self.forecasting_models[metric]['linear_trend'] = lt_model
                        progress.update(metric_task, advance=1)
                        progress.update(overall_task, advance=1)

                    # XGBoost
                    if train_xgboost:
                        progress.update(metric_task, description=f"  [yellow]{metric}[/yellow] → XGBoost")
                        xgb_model = XGBoostForecaster(
                            config=self.config,
                            validation_months=xgboost_validation_months,
                            cv_folds=xgboost_cv_folds
                        )
                        xgb_model.fit(df_train, metric)
                        self.forecasting_models[metric]['xgboost'] = xgb_model
                        progress.update(metric_task, advance=1)
                        progress.update(overall_task, advance=1)

                    progress.update(metric_task, description=f"  [green]✓ {metric}")
                    progress.remove_task(metric_task)
        else:
            # Fallback to basic logging (no rich available)
            model_count = 0
            for metric_idx, metric in enumerate(available_metrics, 1):
                logger.info(f"\n{'=' * 60}")
                logger.info(f"Training models for: {metric} [{metric_idx}/{len(available_metrics)}]")
                logger.info(f"{'=' * 60}")

                self.forecasting_models[metric] = {}

                # Baseline models
                if train_baseline:
                    logger.info(f"[{model_count + 1}/{total_models}] Training Seasonal Naive...")
                    sn_model = SeasonalNaiveForecaster()
                    sn_model.fit(df_train, metric)
                    self.forecasting_models[metric]['seasonal_naive'] = sn_model
                    model_count += 1

                    logger.info(f"[{model_count + 1}/{total_models}] Training Moving Average...")
                    ma_model = MovingAverageForecaster(window=3)
                    ma_model.fit(df_train, metric)
                    self.forecasting_models[metric]['moving_average'] = ma_model
                    model_count += 1

                    logger.info(f"[{model_count + 1}/{total_models}] Training Linear Trend...")
                    lt_model = LinearTrendForecaster()
                    lt_model.fit(df_train, metric)
                    self.forecasting_models[metric]['linear_trend'] = lt_model
                    model_count += 1

                # XGBoost
                if train_xgboost:
                    logger.info(f"[{model_count + 1}/{total_models}] Training XGBoost...")
                    xgb_model = XGBoostForecaster(
                        config=self.config,
                        validation_months=xgboost_validation_months,
                        cv_folds=xgboost_cv_folds
                    )
                    xgb_model.fit(df_train, metric)
                    self.forecasting_models[metric]['xgboost'] = xgb_model
                    model_count += 1

        # Train revenue models
        if train_revenue and 'total_revenue_all' in df_train.columns:
            logger.info(f"\n{'=' * 60}")
            logger.info("Training Revenue Percentage Models")
            logger.info(f"{'=' * 60}")

            # Simple percentage model
            logger.info("Training Simple Percentage Model...")
            percentage_model = RevenuePercentageModel()
            percentage_model.fit(df_train)

            # ML-based model
            logger.info("Training ML Percentage Model...")
            ml_model = RevenueMLModel(config=self.config)
            ml_model.fit(df_train)

            # Ensemble
            logger.info("Creating Revenue Ensemble...")
            self.revenue_ensemble = RevenueEnsemble(
                percentage_model=percentage_model,
                ml_model=ml_model,
                config=self.config
            )

            logger.info(f"✅ Revenue ensemble weights: {self.revenue_ensemble.get_weights()}")

        # Save models if requested
        if save_models:
            self._save_models()

        logger.info("=" * 80)
        logger.info("✅ ALL MODELS TRAINED SUCCESSFULLY")
        logger.info("=" * 80)

        return self.forecasting_models

    def generate_forecast(
        self,
        year: int,
        n_months: int = 12,
        model_type: str = 'xgboost',
        include_revenue_forecast: bool = True,
        save_forecast: bool = True
    ) -> pd.DataFrame:
        """
        Generate forecasts for specified period

        Args:
            year: Year to forecast
            n_months: Number of months to forecast
            model_type: Model to use ('xgboost', 'seasonal_naive', 'moving_average', 'linear_trend')
            include_revenue_forecast: Include total revenue forecast
            save_forecast: Save forecast to CSV

        Returns:
            DataFrame with forecasts
        """
        logger.info("=" * 80)
        logger.info(f"GENERATING {n_months}-MONTH FORECAST FOR {year}")
        logger.info("=" * 80)

        if not self.forecasting_models:
            raise ValueError("No models trained. Run train_models() first.")

        # Get core metrics
        core_metrics = self.config.get('features.core_metrics', [
            'revenue_total',
            'personnel_costs',
            'external_drivers'
        ])

        # Generate forecasts for each metric
        forecasts = {}

        for metric in core_metrics:
            if metric not in self.forecasting_models:
                logger.warning(f"⚠️  No model for '{metric}', skipping")
                continue

            if model_type not in self.forecasting_models[metric]:
                logger.warning(f"⚠️  Model type '{model_type}' not available for '{metric}', skipping")
                continue

            logger.info(f"Forecasting {metric} using {model_type}...")

            model = self.forecasting_models[metric][model_type]
            df_forecast = model.predict(
                n_periods=n_months,
                df_history=self.df_time_series
            )

            forecasts[metric] = df_forecast[metric].values

        # Create consolidated forecast dataframe
        if forecasts:
            # Use first metric's dates
            first_metric = list(forecasts.keys())[0]
            first_model = self.forecasting_models[first_metric][model_type]
            df_forecast_base = first_model.predict(n_months, self.df_time_series)

            df_consolidated = pd.DataFrame({
                'date': df_forecast_base['date']
            })

            # Add all metric forecasts
            for metric, values in forecasts.items():
                df_consolidated[metric] = values

            # Add revenue forecast if available
            if include_revenue_forecast and self.revenue_ensemble is not None:
                logger.info("Generating total revenue forecast using ensemble...")

                # Revenue ensemble needs forecast inputs
                df_revenue_input = df_consolidated.copy()

                # Add external_drivers if available
                if 'external_drivers' not in df_revenue_input.columns:
                    df_revenue_input['external_drivers'] = 0

                # Add personnel_costs if available
                if 'personnel_costs' not in df_revenue_input.columns:
                    df_revenue_input['personnel_costs'] = 0

                df_revenue = self.revenue_ensemble.predict(df_revenue_input)

                df_consolidated['total_revenue_all'] = df_revenue['total_revenue_all']
                df_consolidated['revenue_ratio'] = df_revenue['revenue_ratio']
                df_consolidated['revenue_ratio_simple'] = df_revenue['revenue_ratio_simple']
                df_consolidated['revenue_ratio_ml'] = df_revenue['revenue_ratio_ml']

            # Add metadata
            df_consolidated['model_type'] = model_type
            df_consolidated['forecast_date'] = datetime.now().strftime('%Y-%m-%d')

            # Save if requested
            if save_forecast:
                filename = f"forecast_{year}_{model_type}_{datetime.now().strftime('%Y%m%d')}.csv"
                filepath = self.forecasts_dir / filename
                df_consolidated.to_csv(filepath, index=False)
                logger.info(f"✅ Forecast saved to: {filepath}")

            logger.info("=" * 80)
            logger.info("✅ FORECAST GENERATION COMPLETE")
            logger.info("=" * 80)

            return df_consolidated

        else:
            raise ValueError("No forecasts generated")

    def validate_forecast(
        self,
        df_forecast: pd.DataFrame,
        df_actual: pd.DataFrame,
        metrics_to_validate: Optional[List[str]] = None
    ) -> Dict[str, Dict]:
        """
        Validate forecasts against actual data

        Args:
            df_forecast: Forecast data
            df_actual: Actual data
            metrics_to_validate: List of metrics to validate (validates all if None)

        Returns:
            Dictionary of validation results
        """
        logger.info("=" * 80)
        logger.info("VALIDATING FORECASTS")
        logger.info("=" * 80)

        # Get metrics to validate
        if metrics_to_validate is None:
            metrics_to_validate = self.config.get('features.core_metrics', [
                'revenue_total',
                'personnel_costs',
                'external_drivers'
            ])

            # Add total revenue if available
            if 'total_revenue_all' in df_forecast.columns and 'total_revenue_all' in df_actual.columns:
                metrics_to_validate.append('total_revenue_all')

        # Merge forecast and actual on date
        df_forecast['date'] = pd.to_datetime(df_forecast['date'])
        df_actual['date'] = pd.to_datetime(df_actual['date'])

        df_merged = pd.merge(
            df_forecast,
            df_actual,
            on='date',
            suffixes=('_forecast', '_actual')
        )

        logger.info(f"Validating {len(df_merged)} months of forecasts")

        # Calculate metrics
        validation_results = {}

        for metric in metrics_to_validate:
            forecast_col = f"{metric}_forecast"
            actual_col = f"{metric}_actual"

            if forecast_col not in df_merged.columns or actual_col not in df_merged.columns:
                logger.warning(f"⚠️  Metric '{metric}' not available for validation")
                continue

            actual = df_merged[actual_col].values
            predicted = df_merged[forecast_col].values

            # MAPE
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100

            # MAE
            mae = np.mean(np.abs(actual - predicted))

            # RMSE
            rmse = np.sqrt(np.mean((actual - predicted) ** 2))

            # R²
            ss_res = np.sum((actual - predicted) ** 2)
            ss_tot = np.sum((actual - np.mean(actual)) ** 2)
            r2 = 1 - (ss_res / ss_tot)

            validation_results[metric] = {
                'mape': float(mape),
                'mae': float(mae),
                'rmse': float(rmse),
                'r2': float(r2)
            }

            logger.info(f"  {metric}:")
            logger.info(f"    MAPE: {mape:.2f}%")
            logger.info(f"    MAE:  {mae:,.0f}")
            logger.info(f"    RMSE: {rmse:,.0f}")
            logger.info(f"    R²:   {r2:.3f}")

        logger.info("=" * 80)
        logger.info("✅ VALIDATION COMPLETE")
        logger.info("=" * 80)

        return validation_results

    def generate_business_report(
        self,
        df_forecast: pd.DataFrame,
        output_path: Optional[str] = None,
        format: str = 'json'
    ) -> Dict:
        """
        Generate business-friendly forecast report

        Args:
            df_forecast: Forecast data
            output_path: Path to save report (optional)
            format: Report format ('json', 'csv', 'excel')

        Returns:
            Report dictionary
        """
        logger.info("=" * 80)
        logger.info("GENERATING BUSINESS REPORT")
        logger.info("=" * 80)

        # Calculate summary statistics
        report = {
            'forecast_metadata': {
                'forecast_date': datetime.now().strftime('%Y-%m-%d'),
                'model_type': df_forecast['model_type'].iloc[0] if 'model_type' in df_forecast.columns else 'unknown',
                'n_periods': len(df_forecast),
                'start_date': df_forecast['date'].min().strftime('%Y-%m-%d'),
                'end_date': df_forecast['date'].max().strftime('%Y-%m-%d')
            },
            'metrics': {}
        }

        # Core metrics summary
        core_metrics = self.config.get('features.core_metrics', [
            'revenue_total',
            'personnel_costs',
            'external_drivers'
        ])

        for metric in core_metrics:
            if metric not in df_forecast.columns:
                continue

            report['metrics'][metric] = {
                'total': float(df_forecast[metric].sum()),
                'monthly_average': float(df_forecast[metric].mean()),
                'min': float(df_forecast[metric].min()),
                'max': float(df_forecast[metric].max()),
                'std': float(df_forecast[metric].std())
            }

        # Add total revenue if available
        if 'total_revenue_all' in df_forecast.columns:
            report['metrics']['total_revenue_all'] = {
                'total': float(df_forecast['total_revenue_all'].sum()),
                'monthly_average': float(df_forecast['total_revenue_all'].mean()),
                'min': float(df_forecast['total_revenue_all'].min()),
                'max': float(df_forecast['total_revenue_all'].max()),
                'std': float(df_forecast['total_revenue_all'].std())
            }

            # Revenue ratio insights
            if 'revenue_ratio' in df_forecast.columns:
                report['revenue_insights'] = {
                    'average_ratio': float(df_forecast['revenue_ratio'].mean()),
                    'min_ratio': float(df_forecast['revenue_ratio'].min()),
                    'max_ratio': float(df_forecast['revenue_ratio'].max()),
                    'ensemble_weights': self.revenue_ensemble.get_weights() if self.revenue_ensemble else None
                }

        # Save report if path provided
        if output_path:
            output_path = Path(output_path)

            if format == 'json':
                with open(output_path, 'w') as f:
                    json.dump(report, f, indent=2)
                logger.info(f"✅ Report saved to: {output_path}")

            elif format == 'csv':
                # Flatten report for CSV
                df_report = pd.DataFrame([report['metrics']]).T
                df_report.to_csv(output_path)
                logger.info(f"✅ Report saved to: {output_path}")

            elif format == 'excel':
                # Create Excel with multiple sheets
                with pd.ExcelWriter(output_path) as writer:
                    # Metadata sheet
                    df_meta = pd.DataFrame([report['forecast_metadata']])
                    df_meta.to_excel(writer, sheet_name='Metadata', index=False)

                    # Metrics sheet
                    df_metrics = pd.DataFrame(report['metrics']).T
                    df_metrics.to_excel(writer, sheet_name='Metrics Summary')

                    # Full forecast
                    df_forecast.to_excel(writer, sheet_name='Forecast', index=False)

                logger.info(f"✅ Report saved to: {output_path}")

        logger.info("=" * 80)
        logger.info("✅ BUSINESS REPORT COMPLETE")
        logger.info("=" * 80)

        return report

    def _save_models(self):
        """Save trained models to disk"""
        logger.info("Saving models to disk...")

        # Save forecasting models
        for metric, models in self.forecasting_models.items():
            for model_type, model in models.items():
                filename = f"{metric}_{model_type}.pkl"
                filepath = self.models_dir / filename

                with open(filepath, 'wb') as f:
                    pickle.dump(model, f)

                logger.info(f"  Saved: {filename}")

        # Save revenue ensemble
        if self.revenue_ensemble:
            filepath = self.models_dir / "revenue_ensemble.pkl"
            with open(filepath, 'wb') as f:
                pickle.dump(self.revenue_ensemble, f)
            logger.info(f"  Saved: revenue_ensemble.pkl")

        logger.info("✅ All models saved")

    def load_models(self):
        """Load trained models from disk"""
        logger.info("Loading models from disk...")

        # Get core metrics
        core_metrics = self.config.get('features.core_metrics', [
            'revenue_total',
            'personnel_costs',
            'external_drivers'
        ])

        model_types = ['seasonal_naive', 'moving_average', 'linear_trend', 'xgboost']

        # Load forecasting models
        for metric in core_metrics:
            self.forecasting_models[metric] = {}

            for model_type in model_types:
                filename = f"{metric}_{model_type}.pkl"
                filepath = self.models_dir / filename

                if filepath.exists():
                    with open(filepath, 'rb') as f:
                        model = pickle.load(f)
                    self.forecasting_models[metric][model_type] = model
                    logger.info(f"  Loaded: {filename}")

        # Load revenue ensemble
        filepath = self.models_dir / "revenue_ensemble.pkl"
        if filepath.exists():
            with open(filepath, 'rb') as f:
                self.revenue_ensemble = pickle.load(f)
            logger.info(f"  Loaded: revenue_ensemble.pkl")

        logger.info("✅ All models loaded")
