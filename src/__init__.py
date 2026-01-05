"""
Traveco Forecasting System
==========================

A production-ready forecasting pipeline for transport logistics metrics.

Architecture:
    - src/models/       - Forecasting models (Prior Year, Prophet, SARIMAX, XGBoost)
    - src/operational/  - Operational metrics pipeline
    - src/financial/    - Financial metrics pipeline
    - src/dashboard.py  - Dashboard generation (reads single output file)

Usage:
    python run_pipeline.py  # Run complete pipeline
"""

__version__ = "2.0.0"
__author__ = "Traveco Forecasting Team"
