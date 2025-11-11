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

from src.data.loaders import DataLoader
from src.data.cleaners import DataCleaner
from src.data.validators import DataValidator
from src.data.aggregators import DataAggregator
from src.features.engineering import FeatureEngine
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

    def load_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        validate: bool = True
    ) -> pd.DataFrame:
        """
        Load and prepare historical data

        Args:
            start_date: Start date for filtering (YYYY-MM-DD format)
            end_date: End date for filtering (YYYY-MM-DD format)
            validate: Whether to run data validation

        Returns:
            DataFrame with complete time series
        """
        logger.info("=" * 80)
        logger.info("LOADING HISTORICAL DATA")
        logger.info("=" * 80)

        # Load all data sources
        logger.info("Loading orders data...")
        df_orders = self.loader.load_orders()

        logger.info("Loading tours data...")
        df_tours = self.loader.load_tours()

        logger.info("Loading working days data...")
        df_working_days = self.loader.load_working_days()

        # Load NEW data sources if available
        try:
            logger.info("Loading personnel costs data...")
            df_personnel = self.loader.load_personnel_costs()
        except Exception as e:
            logger.warning(f"Personnel costs data not available: {e}")
            df_personnel = None

        try:
            logger.info("Loading total revenue data...")
            df_total_revenue = self.loader.load_total_revenue()
        except Exception as e:
            logger.warning(f"Total revenue data not available: {e}")
            df_total_revenue = None

        # Clean data
        logger.info("Cleaning orders data...")
        df_orders_clean = self.cleaner.clean_orders(df_orders)

        logger.info("Cleaning tours data...")
        df_tours_clean = self.cleaner.clean_tours(df_tours)

        # Validate if requested
        if validate:
            logger.info("Validating data quality...")

            validation_report = self.validator.generate_validation_report(
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
        save_models: bool = True
    ) -> Dict[str, Dict]:
        """
        Train all forecasting models

        Args:
            df_train: Training data (uses cached data if None)
            train_baseline: Train baseline models
            train_xgboost: Train XGBoost models
            train_revenue: Train revenue models
            save_models: Save trained models to disk

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

        logger.info(f"Training models for {len(core_metrics)} core metrics: {core_metrics}")

        # Train models for each metric
        for metric in core_metrics:
            if metric not in df_train.columns:
                logger.warning(f"⚠️  Metric '{metric}' not found in training data, skipping")
                continue

            logger.info(f"\n{'=' * 60}")
            logger.info(f"Training models for: {metric}")
            logger.info(f"{'=' * 60}")

            self.forecasting_models[metric] = {}

            # Baseline models
            if train_baseline:
                # Seasonal Naive
                logger.info("Training Seasonal Naive...")
                sn_model = SeasonalNaiveForecaster()
                sn_model.fit(df_train, metric)
                self.forecasting_models[metric]['seasonal_naive'] = sn_model

                # Moving Average
                logger.info("Training Moving Average...")
                ma_model = MovingAverageForecaster(window=3)
                ma_model.fit(df_train, metric)
                self.forecasting_models[metric]['moving_average'] = ma_model

                # Linear Trend
                logger.info("Training Linear Trend...")
                lt_model = LinearTrendForecaster()
                lt_model.fit(df_train, metric)
                self.forecasting_models[metric]['linear_trend'] = lt_model

            # XGBoost
            if train_xgboost:
                logger.info("Training XGBoost...")
                xgb_model = XGBoostForecaster(config=self.config)
                xgb_model.fit(df_train, metric)
                self.forecasting_models[metric]['xgboost'] = xgb_model

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
