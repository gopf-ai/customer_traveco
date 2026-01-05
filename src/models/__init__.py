"""Forecasting models for Traveco pipeline"""

from .base import BaseForecaster
from .prior_year import PriorYearForecaster
from .prophet_model import ProphetForecaster
from .sarimax_model import SARIMAXForecaster
from .xgboost_forecaster import XGBoostForecaster
from .baseline_forecasters import SeasonalNaiveForecaster, MovingAverageForecaster, LinearTrendForecaster

__all__ = [
    'BaseForecaster',
    'PriorYearForecaster',
    'ProphetForecaster',
    'SARIMAXForecaster',
    'XGBoostForecaster',
    'SeasonalNaiveForecaster',
    'MovingAverageForecaster',
    'LinearTrendForecaster'
]


# Model registry for easy lookup by name
MODEL_REGISTRY = {
    'prior_year': PriorYearForecaster,
    'prophet': ProphetForecaster,
    'sarimax': SARIMAXForecaster,
    'xgboost': XGBoostForecaster,
    'seasonal_naive': SeasonalNaiveForecaster,
    'moving_average': MovingAverageForecaster,
    'linear_trend': LinearTrendForecaster
}


def get_model(model_name: str, **kwargs):
    """
    Get a model instance by name

    Args:
        model_name: Name of the model (prior_year, prophet, sarimax, xgboost)
        **kwargs: Arguments to pass to model constructor

    Returns:
        Model instance
    """
    model_name = model_name.lower().replace(' ', '_').replace('-', '_')

    if model_name not in MODEL_REGISTRY:
        available = ', '.join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")

    return MODEL_REGISTRY[model_name](**kwargs)
