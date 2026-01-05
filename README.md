# Traveco Forecast Dashboard

Interactive forecasting dashboard for Traveco Transporte AG (Swiss logistics company).

## Overview

This project generates an interactive HTML dashboard with ML-based forecasts for operational and financial metrics. The dashboard displays historical data (2022-Nov 2025) and forecasts (Dec 2025-Dec 2026).

**Output**: `results/forecast_dashboard_2025_2026.html`

## Quick Start

```bash
# Generate the dashboard
pipenv run python scripts/create_forecast_visualization.py

# Open in browser
open results/forecast_dashboard_2025_2026.html
```

To regenerate all forecasts from scratch:

```bash
pipenv run python scripts/extract_financial_metrics.py
pipenv run python scripts/forecast_financial_metrics.py
pipenv run python scripts/generate_operational_forecasts.py
pipenv run python scripts/generate_operational_comparison.py
pipenv run python scripts/create_forecast_visualization.py
```

## Dashboard Features

### Three Tabs
- **Diagramme** - Interactive charts with multiple forecast models
- **Datentabelle** - Monthly data table with yearly summary
- **Information** - German glossary explaining ML models and MAPE

### Metrics

| Metric | Type | Best Model | MAPE |
|--------|------|------------|------|
| Aufträge | Operational | XGBoost | 3.60% |
| Umsatz (Operativ) | Operational | Seasonal Naive | 4.10% |
| Total Betriebsertrag | Financial | XGBoost | 3.06% |
| Betriebsertrag | Financial | XGBoost | 3.22% |
| Personalaufwand | Financial | XGBoost | 2.98% |
| Ausgangsfrachten LKW | Financial | Prior Year | 6.01% |
| EBT | Financial | XGBoost | 92.89% |

### Chart Interaction
- Hover over data points to see values and working days
- Click legend items to show/hide comparison models
- Best model shown by default, others hidden

## Project Structure

```
customer_traveco/
├── scripts/                    # Pipeline scripts
│   ├── extract_financial_metrics.py
│   ├── forecast_financial_metrics.py
│   ├── generate_operational_forecasts.py
│   ├── generate_operational_comparison.py
│   └── create_forecast_visualization.py
├── data/
│   ├── raw/                    # Input Excel files
│   └── processed/              # Generated CSV files
├── results/
│   └── forecast_dashboard_2025_2026.html
├── CLAUDE.md                   # Development guide
└── README.md                   # This file
```

## Technology

- Python 3.10+
- XGBoost, Prophet (forecasting)
- Plotly (interactive charts)
- pandas (data processing)

## Documentation

See `CLAUDE.md` for detailed development guide.

---

**Client**: Traveco Transporte AG
**Last Updated**: January 2026
