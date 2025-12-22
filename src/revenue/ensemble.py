"""Ensemble revenue modeling combining simple and ML approaches"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from src.revenue.percentage_model import RevenuePercentageModel
from src.revenue.ml_model import RevenueMLModel
from src.utils.config import ConfigLoader
from src.utils.logging_config import get_logger


logger = get_logger(__name__)


class RevenueEnsemble:
    """
    Ensemble revenue model

    Combines simple percentage model and ML model with dynamic weighting
    based on validation performance.

    Weighting strategy:
    - If ML MAPE < 3%: Use 70% ML + 30% Simple (ML is excellent)
    - If ML MAPE > 10%: Use 100% Simple (ML is poor)
    - Otherwise: Use 50-50 weighted average
    """

    def __init__(
        self,
        percentage_model: RevenuePercentageModel,
        ml_model: RevenueMLModel,
        config: Optional[ConfigLoader] = None
    ):
        """
        Initialize Revenue Ensemble

        Args:
            percentage_model: Fitted percentage model
            ml_model: Fitted ML model
            config: Configuration loader instance
        """
        self.percentage_model = percentage_model
        self.ml_model = ml_model
        self.config = config if config else ConfigLoader()

        self.weights = None
        self._determine_weights()

    def _determine_weights(self):
        """
        Determine ensemble weights based on ML validation performance
        """
        # Get thresholds from config
        threshold_excellent = self.config.get('revenue_modeling.ml_mape_threshold_excellent', 3.0)
        threshold_poor = self.config.get('revenue_modeling.ml_mape_threshold_poor', 10.0)

        # Get default weights
        default_weights = self.config.get('revenue_modeling.default_weights', {
            'percentage_model': 0.3,
            'ml_model': 0.7
        })

        # Check ML model validation performance
        ml_mape = self.ml_model.validation_mape

        if ml_mape is None:
            logger.warning("ML model validation MAPE not available, using default weights")
            self.weights = default_weights

        elif ml_mape < threshold_excellent:
            # ML is excellent, use mostly ML
            self.weights = {'percentage_model': 0.2, 'ml_model': 0.8}
            logger.info(f"✨ ML model is excellent (MAPE={ml_mape:.2f}%), using 80% ML")

        elif ml_mape > threshold_poor:
            # ML is poor, use only simple
            self.weights = {'percentage_model': 1.0, 'ml_model': 0.0}
            logger.warning(f"⚠️  ML model is poor (MAPE={ml_mape:.2f}%), using 100% Simple")

        else:
            # ML is okay, use balanced
            self.weights = {'percentage_model': 0.5, 'ml_model': 0.5}
            logger.info(f"ML model is okay (MAPE={ml_mape:.2f}%), using 50-50 weights")

    def predict(self, df_forecast: pd.DataFrame) -> pd.DataFrame:
        """
        Generate ensemble predictions

        Args:
            df_forecast: DataFrame with columns [date, revenue_total, external_drivers, ...]

        Returns:
            DataFrame with columns [date, revenue_total, revenue_ratio, total_revenue_all,
                                   revenue_ratio_simple, revenue_ratio_ml]
        """
        logger.info(f"Generating ensemble revenue predictions for {len(df_forecast)} periods")

        # Get predictions from both models
        df_simple = self.percentage_model.predict(df_forecast.copy())
        df_ml = self.ml_model.predict(df_forecast.copy())

        # Create output dataframe
        df_ensemble = df_forecast.copy()

        # Store individual model predictions
        df_ensemble['revenue_ratio_simple'] = df_simple['revenue_ratio']
        df_ensemble['revenue_ratio_ml'] = df_ml['revenue_ratio']

        # Calculate weighted ensemble ratio
        df_ensemble['revenue_ratio'] = (
            self.weights['percentage_model'] * df_ensemble['revenue_ratio_simple'] +
            self.weights['ml_model'] * df_ensemble['revenue_ratio_ml']
        )

        # Ensure ratio >= 1.0
        df_ensemble['revenue_ratio'] = df_ensemble['revenue_ratio'].clip(lower=1.0)

        # Calculate total revenue
        df_ensemble['total_revenue_all'] = df_ensemble['revenue_total'] * df_ensemble['revenue_ratio']

        # Log summary
        logger.info(f"  Ensemble weights: Simple={self.weights['percentage_model']:.1%}, ML={self.weights['ml_model']:.1%}")
        logger.info(f"  Ratio range: {df_ensemble['revenue_ratio'].min():.3f} - {df_ensemble['revenue_ratio'].max():.3f}")
        logger.info(f"  Mean ratio: {df_ensemble['revenue_ratio'].mean():.3f}")
        logger.info(f"  Total revenue: CHF {df_ensemble['total_revenue_all'].sum():,.0f}")

        return df_ensemble

    def validate(
        self,
        df_actual: pd.DataFrame
    ) -> Dict[str, Dict]:
        """
        Validate all models (simple, ML, ensemble) on actual data

        Args:
            df_actual: DataFrame with columns [date, revenue_total, total_revenue_all, ...]

        Returns:
            Dictionary with validation metrics for each model
        """
        logger.info("Validating Revenue Ensemble")

        # Validate simple model
        simple_metrics = self.percentage_model.validate(df_actual)

        # Validate ML model
        ml_metrics = self.ml_model.validate(df_actual)

        # Validate ensemble
        df_ensemble_pred = self.predict(df_actual[['date', 'revenue_total', 'external_drivers'] +
                                                  (['personnel_costs'] if 'personnel_costs' in df_actual.columns else [])])

        actual = df_actual['total_revenue_all'].values
        predicted = df_ensemble_pred['total_revenue_all'].values

        # MAPE
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100

        # MAE
        mae = np.mean(np.abs(actual - predicted))

        # RMSE
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))

        ensemble_metrics = {
            'mape': float(mape),
            'mae': float(mae),
            'rmse': float(rmse)
        }

        # Log comparison
        logger.info("  Validation Results:")
        logger.info(f"    Simple:   MAPE={simple_metrics['mape']:.2f}%")
        logger.info(f"    ML:       MAPE={ml_metrics['mape']:.2f}%")
        logger.info(f"    Ensemble: MAPE={ensemble_metrics['mape']:.2f}%")

        # Determine best model
        best_model = min(
            [('Simple', simple_metrics['mape']),
             ('ML', ml_metrics['mape']),
             ('Ensemble', ensemble_metrics['mape'])],
            key=lambda x: x[1]
        )

        logger.info(f"  🏆 Best model: {best_model[0]} (MAPE={best_model[1]:.2f}%)")

        return {
            'simple': simple_metrics,
            'ml': ml_metrics,
            'ensemble': ensemble_metrics,
            'best_model': best_model[0],
            'weights': self.weights
        }

    def get_model_comparison(self, df_forecast: pd.DataFrame) -> pd.DataFrame:
        """
        Get comparison of all three models

        Args:
            df_forecast: DataFrame with forecast inputs

        Returns:
            DataFrame with predictions from all models
        """
        # Get predictions from each model
        df_simple = self.percentage_model.predict(df_forecast.copy())
        df_ml = self.ml_model.predict(df_forecast.copy())
        df_ensemble = self.predict(df_forecast.copy())

        # Create comparison dataframe
        df_comparison = pd.DataFrame({
            'date': df_ensemble['date'],
            'revenue_total': df_ensemble['revenue_total'],
            'ratio_simple': df_simple['revenue_ratio'],
            'ratio_ml': df_ml['revenue_ratio'],
            'ratio_ensemble': df_ensemble['revenue_ratio'],
            'total_revenue_simple': df_simple['total_revenue_all'],
            'total_revenue_ml': df_ml['total_revenue_all'],
            'total_revenue_ensemble': df_ensemble['total_revenue_all']
        })

        return df_comparison

    def get_weights(self) -> Dict[str, float]:
        """
        Get current ensemble weights

        Returns:
            Dictionary with model weights
        """
        return self.weights.copy()
