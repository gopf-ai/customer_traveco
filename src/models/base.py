"""Base forecaster class for all forecasting models"""

from abc import ABC, abstractmethod
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, Optional
from src.utils.logging_config import get_logger


logger = get_logger(__name__)


class BaseForecaster(ABC):
    """
    Abstract base class for all forecasting models

    All forecasters must implement:
    - fit(): Train the model
    - predict(): Generate forecasts
    - validate(): Calculate validation metrics
    """

    def __init__(self, model_name: str):
        """
        Initialize forecaster

        Args:
            model_name: Name of the model (for logging and saving)
        """
        self.model_name = model_name
        self.model = None
        self.is_fitted = False
        self.training_metrics = {}
        self.validation_metrics = {}

    @abstractmethod
    def fit(self, df_train: pd.DataFrame, target_col: str) -> 'BaseForecaster':
        """
        Train the model

        Args:
            df_train: Training data
            target_col: Name of target column to forecast

        Returns:
            Self (for method chaining)
        """
        pass

    @abstractmethod
    def predict(
        self,
        n_periods: int,
        df_history: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Generate forecasts for n periods ahead

        Args:
            n_periods: Number of periods to forecast
            df_history: Historical data for context (required for some models)

        Returns:
            DataFrame with forecasts
        """
        pass

    def validate(
        self,
        df_val: pd.DataFrame,
        target_col: str
    ) -> Dict[str, float]:
        """
        Validate model on holdout data

        Args:
            df_val: Validation data
            target_col: Name of target column

        Returns:
            Dictionary of validation metrics
        """
        logger.info(f"Validating {self.model_name} on {len(df_val)} samples")

        # Generate predictions
        # Note: This assumes df_val has the necessary features
        predictions = self.predict(len(df_val), df_val)

        # Calculate metrics
        actuals = df_val[target_col].values
        predicted = predictions[target_col].values

        metrics = self._calculate_metrics(actuals, predicted)

        self.validation_metrics = metrics

        logger.info(f"  MAPE: {metrics['mape']:.2f}%")
        logger.info(f"  MAE:  {metrics['mae']:,.2f}")

        return metrics

    def _calculate_metrics(
        self,
        actuals: pd.Series,
        predicted: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate forecast accuracy metrics

        Args:
            actuals: Actual values
            predicted: Predicted values

        Returns:
            Dictionary with metrics
        """
        import numpy as np

        # Remove NaN values
        mask = ~(pd.isna(actuals) | pd.isna(predicted))
        actuals = actuals[mask]
        predicted = predicted[mask]

        if len(actuals) == 0:
            return {
                'mape': np.nan,
                'mae': np.nan,
                'rmse': np.nan,
                'r2': np.nan
            }

        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((actuals - predicted) / actuals)) * 100

        # MAE (Mean Absolute Error)
        mae = np.mean(np.abs(actuals - predicted))

        # RMSE (Root Mean Squared Error)
        rmse = np.sqrt(np.mean((actuals - predicted) ** 2))

        # R² (Coefficient of Determination)
        ss_res = np.sum((actuals - predicted) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

        return {
            'mape': float(mape),
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2)
        }

    def save(self, path: str):
        """
        Save model to disk

        Args:
            path: Path to save model (e.g., 'models/trained/xgboost_revenue.pkl')
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump(self, f)

        logger.info(f"Model saved to: {path}")

    @classmethod
    def load(cls, path: str) -> 'BaseForecaster':
        """
        Load model from disk

        Args:
            path: Path to saved model

        Returns:
            Loaded model instance
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, 'rb') as f:
            model = pickle.load(f)

        logger.info(f"Model loaded from: {path}")

        return model

    def get_metadata(self) -> Dict:
        """
        Get model metadata

        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'is_fitted': self.is_fitted,
            'training_metrics': self.training_metrics,
            'validation_metrics': self.validation_metrics
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name='{self.model_name}', is_fitted={self.is_fitted})"
