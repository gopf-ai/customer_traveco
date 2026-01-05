"""Operational forecast generation pipeline

Uses best model per metric to generate forecasts.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from src.models import get_model
from src.utils.logging_config import get_logger
from src.utils.config import ConfigLoader


logger = get_logger(__name__)


# Metrics that benefit from working_days feature
METRICS_WITH_WORKING_DAYS = [
    'total_orders',
    'total_km_billed',
    'total_km_actual',
    'total_tours',
    'total_drivers',
    'revenue_total',
    'vehicle_km_cost',
    'vehicle_time_cost',
    'total_vehicle_cost'
]


class OperationalForecast:
    """
    Generate operational forecasts using best model per metric

    Steps:
    1. Load time series and evaluation results
    2. For each metric, retrain best model on all data
    3. Generate forecasts for horizon
    4. Combine actuals + forecasts
    5. Save to output directory
    """

    def __init__(self, config: Optional[ConfigLoader] = None):
        """Initialize forecast generation"""
        self.config = config if config else ConfigLoader()
        self.intermediate_path = Path(self.config.get('data.intermediate_path', 'data/intermediate'))
        self.output_path = Path(self.config.get('data.output_path', 'data/output'))

        if not self.intermediate_path.is_absolute():
            project_root = Path(__file__).parent.parent.parent
            self.intermediate_path = project_root / self.intermediate_path

        if not self.output_path.is_absolute():
            project_root = Path(__file__).parent.parent.parent
            self.output_path = project_root / self.output_path

        self.output_path.mkdir(parents=True, exist_ok=True)

        # Forecast horizon
        self.forecast_start = self.config.get('forecast.start', '2025-12-01')
        self.forecast_end = self.config.get('forecast.end', '2026-12-31')

    def run(self) -> pd.DataFrame:
        """
        Run the complete forecast generation pipeline

        Returns:
            DataFrame with actuals and forecasts
        """
        logger.info("=" * 60)
        logger.info("OPERATIONAL FORECAST GENERATION")
        logger.info("=" * 60)

        # Step 1: Load data
        df_ts = self._load_time_series()
        df_eval = self._load_eval_results()

        # Step 2: Get best models
        best_models = self._get_best_models(df_eval)

        # Step 3: Generate forecasts for each metric
        all_forecasts = []

        for metric, model_name in best_models.items():
            logger.info(f"\n--- Forecasting {metric} with {model_name} ---")

            try:
                df_forecast = self._generate_metric_forecast(
                    df_ts, metric, model_name
                )
                all_forecasts.append(df_forecast)

            except Exception as e:
                logger.error(f"Error forecasting {metric}: {e}")

        # Step 4: Combine all forecasts
        df_combined = self._combine_forecasts(df_ts, all_forecasts, best_models)

        # Step 5: Save output
        output_file = self.output_path / 'operational_forecasts.csv'
        df_combined.to_csv(output_file, index=False)
        logger.info(f"\nSaved operational forecasts to: {output_file}")

        return df_combined

    def _load_time_series(self) -> pd.DataFrame:
        """Load operational time series"""
        input_path = self.intermediate_path / 'operational_time_series.csv'

        if not input_path.exists():
            raise FileNotFoundError(f"Time series file not found: {input_path}")

        df = pd.read_csv(input_path)
        df['date'] = pd.to_datetime(df['date'])

        logger.info(f"Loaded time series: {len(df)} months")

        return df

    def _load_eval_results(self) -> pd.DataFrame:
        """Load model evaluation results"""
        input_path = self.intermediate_path / 'operational_model_eval.csv'

        if not input_path.exists():
            raise FileNotFoundError(f"Evaluation results not found: {input_path}")

        df = pd.read_csv(input_path)
        logger.info(f"Loaded evaluation results: {len(df)} model-metric combinations")

        return df

    def _get_best_models(self, df_eval: pd.DataFrame) -> Dict[str, str]:
        """Extract best model per metric from evaluation results"""
        best_models = {}

        best_rows = df_eval[df_eval['is_best'] == True]

        for _, row in best_rows.iterrows():
            metric = row['metric']
            model = row['model']
            mape = row['mape']

            best_models[metric] = model
            logger.info(f"Best model for {metric}: {model} (MAPE: {mape:.2f}%)")

        return best_models

    def _generate_metric_forecast(
        self,
        df_ts: pd.DataFrame,
        metric: str,
        model_name: str
    ) -> pd.DataFrame:
        """Generate forecast for a single metric"""
        # Check if metric uses working_days
        use_working_days = (
            metric in METRICS_WITH_WORKING_DAYS and
            'working_days' in df_ts.columns
        )

        # Initialize model
        if model_name == 'prophet':
            model = get_model(model_name, include_working_days=use_working_days)
        elif model_name == 'sarimax':
            model = get_model(model_name, include_working_days=use_working_days)
        else:
            model = get_model(model_name)

        # Train on all available data
        logger.info(f"Training {model_name} on {len(df_ts)} months...")
        model.fit(df_ts, metric)

        # Calculate forecast periods
        last_actual_date = df_ts['date'].max()
        forecast_start = pd.to_datetime(self.forecast_start)
        forecast_end = pd.to_datetime(self.forecast_end)

        # Number of periods from last actual to forecast end
        n_periods = (forecast_end.year - last_actual_date.year) * 12 + \
                   (forecast_end.month - last_actual_date.month)

        logger.info(f"Generating {n_periods}-month forecast...")

        # Generate forecast
        df_forecast = model.predict(n_periods, df_ts)

        # Filter to desired range
        df_forecast = df_forecast[
            (df_forecast['date'] >= forecast_start) &
            (df_forecast['date'] <= forecast_end)
        ]

        # Add metadata
        df_forecast['model'] = model_name
        df_forecast['metric_name'] = metric

        # Get MAPE from training metrics
        df_forecast['model_mape'] = model.training_metrics.get('mape', np.nan)

        return df_forecast

    def _combine_forecasts(
        self,
        df_ts: pd.DataFrame,
        forecasts: list,
        best_models: Dict[str, str]
    ) -> pd.DataFrame:
        """Combine actuals and forecasts into single output"""
        records = []

        # Add actuals (long format)
        for metric in best_models.keys():
            if metric not in df_ts.columns:
                continue

            for _, row in df_ts.iterrows():
                records.append({
                    'date': row['date'],
                    'metric': metric,
                    'value': row[metric],
                    'source': 'actual',
                    'model': None,
                    'mape': None,
                    'pipeline': 'operational'
                })

        # Add forecasts (already in long-ish format)
        for df_forecast in forecasts:
            metric = df_forecast['metric_name'].iloc[0]
            model = df_forecast['model'].iloc[0]
            mape = df_forecast['model_mape'].iloc[0]

            for _, row in df_forecast.iterrows():
                records.append({
                    'date': row['date'],
                    'metric': metric,
                    'value': row[metric],
                    'source': 'forecast',
                    'model': model,
                    'mape': mape,
                    'pipeline': 'operational'
                })

        df_combined = pd.DataFrame(records)
        df_combined = df_combined.sort_values(['metric', 'date']).reset_index(drop=True)

        logger.info(f"Combined output: {len(df_combined)} records")
        logger.info(f"  Actuals: {len(df_combined[df_combined['source'] == 'actual'])}")
        logger.info(f"  Forecasts: {len(df_combined[df_combined['source'] == 'forecast'])}")

        return df_combined


def run():
    """Entry point for pipeline"""
    forecast = OperationalForecast()
    return forecast.run()


if __name__ == '__main__':
    run()
