"""Financial forecasting pipeline"""

from .data_prep import FinancialDataPrep, run as run_data_prep
from .model_eval import FinancialModelEval, run as run_model_eval
from .forecast import FinancialForecast, run as run_forecast


def run_pipeline():
    """Run the complete financial pipeline"""
    # Step 1: Data preparation
    run_data_prep()

    # Step 2: Model evaluation
    run_model_eval()

    # Step 3: Generate forecasts
    run_forecast()


__all__ = [
    'FinancialDataPrep',
    'FinancialModelEval',
    'FinancialForecast',
    'run_data_prep',
    'run_model_eval',
    'run_forecast',
    'run_pipeline'
]
