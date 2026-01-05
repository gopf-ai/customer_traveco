# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Traveco Forecast Dashboard** - A production-ready forecasting system for Traveco Transporte AG, a Swiss transport/logistics company. The system generates an interactive HTML dashboard with ML-based forecasts for operational and financial metrics.

**Client**: Traveco (Switzerland)
**Output**: `results/forecast_dashboard_2025_2026.html`
**Last Updated**: January 2026 (data through November 2025)

## Quick Start

```bash
# Generate the complete dashboard (run all scripts in sequence)
pipenv run python scripts/extract_financial_metrics.py
pipenv run python scripts/forecast_financial_metrics.py
pipenv run python scripts/generate_operational_forecasts.py
pipenv run python scripts/generate_operational_comparison.py
pipenv run python scripts/create_forecast_visualization.py

# Open the dashboard
open results/forecast_dashboard_2025_2026.html
```

## Architecture

The project uses a **scripts-based pipeline** that processes raw data and generates an interactive HTML dashboard:

```
Raw Data (Excel) → Scripts → Processed CSV → Dashboard (HTML)
```

### Key Scripts

| Script | Purpose |
|--------|---------|
| `extract_financial_metrics.py` | Extract metrics from yearly financial Excel files |
| `forecast_financial_metrics.py` | Train XGBoost/Prophet models for financial metrics |
| `generate_operational_forecasts.py` | Generate operational metric forecasts |
| `generate_operational_comparison.py` | Generate comparison forecasts for operational metrics |
| `create_forecast_visualization.py` | Generate interactive HTML dashboard |

## Repository Structure

```
customer_traveco/
├── scripts/                              # Main pipeline scripts
│   ├── extract_financial_metrics.py      # Financial data extraction
│   ├── forecast_financial_metrics.py     # Financial forecasting (XGBoost/Prophet)
│   ├── generate_operational_forecasts.py # Operational forecasting
│   ├── generate_operational_comparison.py# Comparison forecasts with working days
│   └── create_forecast_visualization.py  # Dashboard generation
├── src/                                  # Source modules
│   ├── data/loaders.py                   # Data loading utilities
│   ├── models/                           # Model implementations
│   ├── financial/                        # Financial metric processing
│   └── operational/                      # Operational metric processing
├── data/
│   ├── raw/                              # Raw Excel files from Traveco
│   │   ├── TRAVECO_Arbeitstage_*.xlsx    # Working days per month (2022-2026)
│   │   └── swisstransfer_*/              # Order and tour data
│   └── processed/                        # Generated CSV files
│       ├── financial_metrics_*.csv       # Financial actuals and forecasts
│       ├── monthly_aggregated_*.csv      # Operational time series
│       ├── combined_forecast_*.csv       # Combined operational forecasts
│       └── comparison_forecasts.csv      # Multi-model comparison data
├── results/                              # Output files
│   └── forecast_dashboard_2025_2026.html # Main dashboard
├── config/
│   └── config.yaml                       # Configuration settings
├── CLAUDE.md                             # This file
└── README.md                             # Project overview
```

## Dashboard Features

The generated dashboard (`forecast_dashboard_2025_2026.html`) includes:

### Three Tabs
1. **Diagramme** - Interactive charts with multiple forecast models
2. **Datentabelle** - Monthly data table with yearly summary
3. **Information** - German glossary explaining ML models and MAPE

### Metrics Displayed

**Operational Metrics** (from order data):
- Aufträge (Total Orders) - Best: XGBoost (3.60% MAPE)
- Umsatz Operativ (Revenue) - Best: Seasonal Naive (4.10% MAPE)

**Financial Metrics** (from accounting):
- Total Betriebsertrag (SK 0140) - Best: XGBoost (3.06% MAPE)
- Betriebsertrag (SK 3506+3507) - Best: XGBoost (3.22% MAPE)
- Personalaufwand (SK 0151) - Best: XGBoost (2.98% MAPE)
- Ausgangsfrachten LKW (SK 6280) - Best: Prior Year (6.01% MAPE)
- EBT (SK 0110) - Best: XGBoost (92.89% MAPE)

### Chart Features
- **Working days in hover**: Each data point shows the number of working days
- **Multiple models**: Best model shown by default, comparison models hidden
- **Toggle visibility**: Click legend items to show/hide models
- **Model colors**: XGBoost (blue), Prophet (purple), Prior Year (orange), Seasonal Naive (teal)

## Working Days Feature

Working days per month significantly correlate with operational metrics and are used as features in XGBoost models.

**Data Source**: `data/raw/TRAVECO_Arbeitstage_2022-laufend_für gopf.com_hb v2.xlsx`

### Correlation with Metrics

| Metric | Correlation | Used in Model |
|--------|-------------|---------------|
| total_orders | 0.825 | Yes |
| revenue_total | 0.634 | Yes |
| total_revenue (financial) | -0.572 | Yes |
| total_betriebsertrag | -0.568 | Yes |
| ebt | -0.503 | Yes |
| personnel_costs | -0.251 | No (weak) |
| external_driver_costs | 0.203 | No (weak) |

## Model Configuration

### XGBoost
```python
xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)
```

### Prophet
```python
Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    seasonality_mode='multiplicative'
)
```

## Development Commands

```bash
# Run individual scripts
pipenv run python scripts/extract_financial_metrics.py
pipenv run python scripts/forecast_financial_metrics.py
pipenv run python scripts/generate_operational_forecasts.py
pipenv run python scripts/generate_operational_comparison.py
pipenv run python scripts/create_forecast_visualization.py

# Open dashboard
open results/forecast_dashboard_2025_2026.html
```

## Key Data Files

### Input (Raw)
- `TRAVECO_Arbeitstage_*.xlsx` - Working days 2022-2026
- Financial Excel files per year (2022-2025)
- Order/tour data in `swisstransfer_*/`

### Output (Processed)
- `financial_metrics_overview.csv` - Historical financial data
- `financial_metrics_forecasts.csv` - Financial forecasts
- `monthly_aggregated_full_company.csv` - Operational time series
- `combined_forecast_2025_2026.csv` - Operational forecasts
- `comparison_forecasts.csv` - All model predictions for comparison

## Technology Stack

- **Language**: Python 3.10+
- **Environment**: Pipenv (workspace-level at `/Users/kk/dev`)
- **Key Libraries**:
  - pandas, numpy - Data processing
  - xgboost - Gradient boosting models
  - prophet - Time series forecasting
  - plotly - Interactive visualizations
  - openpyxl - Excel file handling

## Domain Context

### Swiss Transport Industry
- **Working days**: Strong correlation with order volume
- **Seasonal patterns**: School holidays, summer peaks
- **Cost structure**: Personnel + external drivers (subcontractors)

### Traveco Terminology
- **Aufträge** - Orders
- **Betriebsertrag** - Operating revenue
- **Personalaufwand** - Personnel costs
- **Ausgangsfrachten** - Outbound freight costs
- **Arbeitstage** - Working days

## Current Status

- **Stage**: Production dashboard system
- **Data Coverage**: 2022-2024 historical + Jan-Nov 2025 actual + Dec 2025-Dec 2026 forecast
- **Key Finding**: ML models outperform traditional methods for orders, but not for revenue
- **Recommendation**: Hybrid approach (ML for operations, traditional for revenue)
