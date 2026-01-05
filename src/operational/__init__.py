"""Operational forecasting pipeline"""

from .data_prep import OperationalDataPrep, run as run_data_prep
from .model_eval import OperationalModelEval, run as run_model_eval
from .forecast import OperationalForecast, run as run_forecast


def run_pipeline():
    """Run the complete operational pipeline"""
    # Step 1: Data preparation
    run_data_prep()

    # Step 2: Model evaluation
    run_model_eval()

    # Step 3: Generate forecasts
    run_forecast()


__all__ = [
    'OperationalDataPrep',
    'OperationalModelEval',
    'OperationalForecast',
    'run_data_prep',
    'run_model_eval',
    'run_forecast',
    'run_pipeline'
]
