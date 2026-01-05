"""Operational model evaluation pipeline

Trains and evaluates all models on operational metrics using train/test split.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from src.models import get_model, MODEL_REGISTRY
from src.utils.logging_config import get_logger
from src.utils.config import ConfigLoader


logger = get_logger(__name__)


# Metrics that benefit from working_days feature (based on correlation analysis)
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


class OperationalModelEval:
    """
    Evaluate forecasting models on operational metrics

    Steps:
    1. Load prepared time series
    2. Split into train/test sets
    3. Train all models on each metric
    4. Evaluate on test set
    5. Save evaluation results
    """

    # Models to evaluate
    MODELS = ['prior_year', 'prophet', 'sarimax', 'xgboost']

    # Metrics to forecast
    METRICS = [
        'total_orders',
        'revenue_total',
        'total_drivers',
        'external_drivers',
        'total_km_billed'
    ]

    def __init__(self, config: Optional[ConfigLoader] = None):
        """Initialize model evaluation"""
        self.config = config if config else ConfigLoader()
        self.intermediate_path = Path(self.config.get('data.intermediate_path', 'data/intermediate'))

        if not self.intermediate_path.is_absolute():
            project_root = Path(__file__).parent.parent.parent
            self.intermediate_path = project_root / self.intermediate_path

        # Train/test split dates
        self.train_end = self.config.get('evaluation.train_end', '2024-12-31')
        self.test_start = self.config.get('evaluation.test_start', '2025-01-01')
        self.test_end = self.config.get('evaluation.test_end', '2025-11-30')

    def run(self) -> pd.DataFrame:
        """
        Run the complete model evaluation pipeline

        Returns:
            DataFrame with evaluation results for all models and metrics
        """
        logger.info("=" * 60)
        logger.info("OPERATIONAL MODEL EVALUATION")
        logger.info("=" * 60)

        # Step 1: Load time series
        df = self._load_time_series()

        # Step 2: Split train/test
        df_train, df_test = self._split_data(df)

        # Step 3: Evaluate all models on all metrics
        results = self._evaluate_all_models(df_train, df_test)

        # Step 4: Select best model per metric
        df_results = pd.DataFrame(results)
        df_results = self._select_best_models(df_results)

        # Step 5: Save results
        output_path = self.intermediate_path / 'operational_model_eval.csv'
        df_results.to_csv(output_path, index=False)
        logger.info(f"Saved evaluation results to: {output_path}")

        # Print summary
        self._print_summary(df_results)

        return df_results

    def _load_time_series(self) -> pd.DataFrame:
        """Load prepared operational time series"""
        input_path = self.intermediate_path / 'operational_time_series.csv'

        if not input_path.exists():
            raise FileNotFoundError(
                f"Time series file not found: {input_path}\n"
                f"Run operational data_prep.py first"
            )

        df = pd.read_csv(input_path)
        df['date'] = pd.to_datetime(df['date'])

        logger.info(f"Loaded time series: {len(df)} months")
        logger.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")

        return df

    def _split_data(self, df: pd.DataFrame) -> tuple:
        """Split data into train and test sets"""
        train_end = pd.to_datetime(self.train_end)
        test_start = pd.to_datetime(self.test_start)
        test_end = pd.to_datetime(self.test_end)

        df_train = df[df['date'] <= train_end].copy()
        df_test = df[(df['date'] >= test_start) & (df['date'] <= test_end)].copy()

        logger.info(f"Train set: {len(df_train)} months ({df_train['date'].min()} to {df_train['date'].max()})")
        logger.info(f"Test set:  {len(df_test)} months ({df_test['date'].min()} to {df_test['date'].max()})")

        return df_train, df_test

    def _evaluate_all_models(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame
    ) -> List[Dict]:
        """Evaluate all models on all metrics"""
        results = []

        for metric in self.METRICS:
            if metric not in df_train.columns:
                logger.warning(f"Metric {metric} not found in data - skipping")
                continue

            logger.info(f"\n--- Evaluating models for: {metric} ---")

            # Check if metric should use working_days
            use_working_days = (
                metric in METRICS_WITH_WORKING_DAYS and
                'working_days' in df_train.columns
            )

            for model_name in self.MODELS:
                try:
                    result = self._evaluate_single_model(
                        model_name, metric, df_train, df_test, use_working_days
                    )
                    results.append(result)

                except Exception as e:
                    logger.error(f"Error evaluating {model_name} for {metric}: {e}")
                    results.append({
                        'metric': metric,
                        'model': model_name,
                        'mape': np.nan,
                        'mae': np.nan,
                        'rmse': np.nan,
                        'r2': np.nan,
                        'error': str(e)
                    })

        return results

    def _evaluate_single_model(
        self,
        model_name: str,
        metric: str,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        use_working_days: bool
    ) -> Dict:
        """Evaluate a single model on a single metric"""
        logger.info(f"  Training {model_name}...")

        # Get model with appropriate configuration
        if model_name == 'prophet':
            model = get_model(model_name, include_working_days=use_working_days)
        elif model_name == 'sarimax':
            model = get_model(model_name, include_working_days=use_working_days)
        elif model_name == 'xgboost':
            model = get_model(model_name)
        else:
            model = get_model(model_name)

        # Train model
        model.fit(df_train, metric)

        # Validate on test set
        metrics = model.validate(df_test, metric)

        return {
            'metric': metric,
            'model': model_name,
            'mape': metrics.get('mape', np.nan),
            'mae': metrics.get('mae', np.nan),
            'rmse': metrics.get('rmse', np.nan),
            'r2': metrics.get('r2', np.nan)
        }

    def _select_best_models(self, df_results: pd.DataFrame) -> pd.DataFrame:
        """Mark best model per metric based on MAPE"""
        df_results['is_best'] = False

        for metric in df_results['metric'].unique():
            metric_mask = df_results['metric'] == metric
            metric_results = df_results[metric_mask]

            # Find best (lowest MAPE)
            valid_results = metric_results[metric_results['mape'].notna()]
            if len(valid_results) > 0:
                best_idx = valid_results['mape'].idxmin()
                df_results.loc[best_idx, 'is_best'] = True

        return df_results

    def _print_summary(self, df_results: pd.DataFrame):
        """Print evaluation summary"""
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 60)

        best_models = df_results[df_results['is_best']]

        for _, row in best_models.iterrows():
            logger.info(f"{row['metric']}: Best = {row['model']} (MAPE: {row['mape']:.2f}%)")


def run():
    """Entry point for pipeline"""
    eval_pipeline = OperationalModelEval()
    return eval_pipeline.run()


if __name__ == '__main__':
    run()
