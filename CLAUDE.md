# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **transport logistics forecasting and analysis project** for Traveco (Travecom), a Swiss transport/logistics company. The project analyzes transport operations and provides ML-based forecasting for 10 key operational metrics.

**Client**: Traveco (Switzerland)
**Objective**: Forecast transport metrics using ML models, validate against actual 2025 data, and provide actionable insights
**Stage**: Production forecasting system with validation framework
**Last Updated**: December 2025 (data through November 2025)
**Key Achievement**: ML forecasts validated against 11 months of 2025 data
- Orders: ML 5.6% more accurate than traditional method
- Revenue: Traditional 7.4% more accurate than ML
- Recommendation: **Hybrid approach** (ML for operations, traditional for revenue)

## ⚠️ CRITICAL CORRECTIONS APPLIED (October 2025)

This project underwent major methodology corrections. Key changes:

1. **Cost Attribution** (MOST CRITICAL):
   - ✅ Aggregated by `Nummer.Auftraggeber` (order owner/payer) - Column G
   - Costs attributed to who pays, not who dispatches
   - Impact: Notebook 04 cells 8-9

2. **Order Type Classification**:
   - ✅ 5 categories: Delivery, Pickup/Multi-leg, Leergut/Empty Returns, Retoure, Abholung
   - Leergut (empty returns) are ~18% of orders
   - Impact: Notebook 03 cell 11

3. **Data Filtering**:
   - ✅ Exclude "Lager Auftrag" (warehouse orders) and missing carrier numbers
   - Impact: `utils/traveco_utils.py`, `config.yaml`

4. **Sparten Mapping**:
   - ✅ Auto-convert to Int64 for reliable matching
   - Impact: `utils/traveco_utils.py` `map_customer_divisions()`

5. **Betriebszentralen Mapping**:
   - ✅ Mapped to 14 Betriebszentralen names (dispatch centers)
   - Impact: `data/raw/TRAVECO_Betriebszentralen.csv`, Notebooks 03-06

6. **Forecasting Metrics Expansion**:
   - ✅ **10 target metrics** (expanded from 5):
     - `total_orders`, `total_drivers`, `revenue_total`, `external_drivers`
     - `total_km_billed` (order-based billing KM)
     - `total_km_actual` (tour-based actual driven KM)
     - `total_tours` (unique tour count)
     - `vehicle_km_cost`, `vehicle_time_cost`, `total_vehicle_cost`
   - **Cost Formula**: `Total Cost = (Actual KM × PC KM Kosten) + (IST Zeit PraCar × 60 × PC Minuten Kosten)`
   - Impact: Notebooks 08-12, 14, 17-18

### Common Errors & Solutions:
- **"AttributeError: map_customer_divisions"**: Restart Jupyter kernel
- **"betriebszentrale_name not found"**: Run Notebook 03 with updated mapping
- **"All orders show 'Keine Sparte'"**: Type mismatch - fixed in latest code

## December 2025 Enhancements

### Financial Metrics Dashboard

Added standalone financial forecasting pipeline with HTML dashboard:

**New Scripts** (`scripts/`):
- `extract_financial_metrics.py` - Extracts metrics from yearly financial Excel files
- `forecast_financial_metrics.py` - Trains Prophet/SARIMAX/XGBoost models
- `create_forecast_visualization.py` - Generates interactive HTML dashboard

**5 Financial Metrics Forecasted**:
| Metric | Sachkonto | Best Model | MAPE |
|--------|-----------|------------|------|
| Total Betriebsertrag | SK 0140 | XGBoost | 3.48% |
| Betriebsertrag | SK 3506+3507 | XGBoost | 2.59% |
| Personalaufwand | SK 0151 | Prophet | 3.19% |
| Ausgangsfrachten LKW | SK 6280 | XGBoost | 5.73% |
| EBT | SK 0110 | XGBoost | 75.49% |

**Dashboard Output**: `results/forecast_dashboard_2025_2026.html`

**Run Pipeline**:
```bash
pipenv run python scripts/extract_financial_metrics.py
pipenv run python scripts/forecast_financial_metrics.py
pipenv run python scripts/create_forecast_visualization.py
```

### Working Days Feature (CFO Insight)

**Key Finding**: Both financial and operational metrics correlate with working days per month.

**Data Source**: `data/raw/TRAVECO_Arbeitstage_2022-laufend_für gopf.com_hb v2.xlsx`

**Correlation Analysis Script**: `scripts/analyze_working_days_correlation.py`
**Output**: `data/processed/working_days_correlation_analysis.csv`

#### Financial Metrics (from `scripts/forecast_financial_metrics.py`)
| Metric | Correlation | Significance | Use Working Days |
|--------|-------------|--------------|------------------|
| total_revenue | -0.572 | p<0.01 | ✅ Yes |
| total_betriebsertrag | -0.568 | p<0.01 | ✅ Yes |
| ebt | -0.503 | p<0.01 | ✅ Yes |
| personnel_costs | -0.251 | weak | ❌ No |
| external_driver_costs | 0.203 | weak | ❌ No |

#### Operational Metrics (from correlation analysis Dec 2025)
| Metric | Correlation | P-Value | Use Working Days |
|--------|-------------|---------|------------------|
| total_drivers | 0.832 | <0.001 | ✅ Yes |
| total_orders | 0.825 | <0.001 | ✅ Yes |
| total_km_billed | 0.804 | <0.001 | ✅ Yes |
| total_tours | 0.710 | <0.001 | ✅ Yes |
| revenue_total | 0.634 | <0.001 | ✅ Yes |
| vehicle_time_cost | 0.615 | <0.001 | ✅ Yes |
| total_vehicle_cost | 0.599 | <0.001 | ✅ Yes |
| total_km_actual | 0.585 | <0.001 | ✅ Yes |
| vehicle_km_cost | 0.562 | <0.001 | ✅ Yes |
| **external_drivers** | **0.278** | **0.10** | ❌ **No** |

**Key Insight**: 9 of 10 operational metrics show strong positive correlation with working days (more working days = more activity). Only `external_drivers` has weak/insignificant correlation and is excluded.

**Implementation**: Notebooks 10 (Prophet) and 12 (XGBoost) now selectively include `working_days` as a feature based on the `METRICS_WITH_WORKING_DAYS` list.

**Note**: EBT MAPE is high because values oscillate around zero (profit/loss). Consider using MAE instead.

### Baseline Method Change

Changed from "2024÷12 fixed monthly" to **"Same month prior year"** baseline for more realistic comparison.

## Technology Stack

- **Language**: Python
- **Primary Environment**: Jupyter Notebooks
- **Key Libraries**:
  - **Analysis**: pandas, numpy, scipy
  - **Visualization**: Plotly (interactive dashboards), matplotlib, seaborn
  - **Presentation**: python-pptx (PowerPoint generation)
  - **Data**: openpyxl (Excel), pyxlsb (.xlsb files)
  - **Forecasting**: Prophet, SARIMAX, XGBoost

## Repository Structure

```
customer_traveco/
├── notebooks/
│   ├── 02_data_cleaning_and_validation.ipynb    # Data cleaning
│   ├── 03_feature_engineering.ipynb              # Features + Sparten + Betriebszentralen
│   ├── 04_aggregation_and_targets.ipynb          # Aggregation by Auftraggeber
│   ├── 05_exploratory_data_analysis.ipynb        # EDA + KM efficiency + Sparten
│   ├── 06_tour_cost_analysis.ipynb               # Vehicle cost calculation
│   ├── 08_time_series_aggregation.ipynb          # Time series prep (10 metrics)
│   ├── 09_baseline_models.ipynb                  # Baseline forecasts
│   ├── 10_prophet_model.ipynb                    # Prophet forecasting
│   ├── 11_sarimax_model.ipynb                    # SARIMAX forecasting
│   ├── 12_xgboost_model.ipynb                    # XGBoost forecasting
│   ├── 14_consolidated_forecasts_2025.ipynb      # Best model selection
│   ├── 15_model_comparison.ipynb                 # Model performance comparison
│   ├── 17_monthly_forecast_2025_table.ipynb      # Forecast table generation
│   └── 18_forecast_validation_2025.ipynb         # Validation vs actual 2025 data
├── data/
│   ├── raw/
│   │   ├── TRAVECO_Betriebszentralen.csv         # 14 dispatch centers
│   │   └── swisstransfer_*/                      # Excel files (orders, tours, Sparten)
│   └── processed/                                 # Processed data files
│       ├── clean_orders.csv
│       ├── features_engineered.csv
│       ├── monthly_aggregated.csv
│       ├── tour_costs.csv
│       ├── time_series_full_company.csv          # 10 metrics time series
│       └── consolidated_forecast_2025.csv        # Final forecasts
├── results/                                       # Generated charts and reports
│   ├── forecast_validation_*.html                # Interactive validation dashboards
│   ├── forecast_validation_*.csv                 # Validation reports
│   ├── forecast_dashboard_2025_2026.html         # Financial metrics dashboard
│   └── FORECAST_MODEL_COMPARISON_FINAL.md        # Model comparison results
├── scripts/                                       # Standalone Python scripts
│   ├── analyze_working_days_correlation.py       # Working days correlation analysis
│   ├── extract_financial_metrics.py              # Financial data extraction
│   ├── forecast_financial_metrics.py             # Financial forecasting
│   └── create_forecast_visualization.py          # Dashboard generation
├── utils/
│   └── traveco_utils.py                          # Core utilities
├── config/
│   └── config.yaml                               # Configuration
├── documentation/                                 # Video recordings
│   └── *.mp4                                      # Session recordings (German)
├── CLAUDE.md                                      # This file - main project guide
├── README.md                                      # Project overview
├── TROUBLESHOOTING.md                             # Common issues and solutions
└── FORECAST_METHODOLOGY.md                        # Technical methodology
```

## Development Commands

### Running Notebooks

```bash
# Start Jupyter from project root
jupyter notebook

# Or use Jupyter Lab
jupyter lab
```

### Notebook Execution Order

```bash
# Phase 1: Data Preparation
notebooks/02_data_cleaning_and_validation.ipynb
notebooks/03_feature_engineering.ipynb
notebooks/04_aggregation_and_targets.ipynb

# Phase 2: Analysis
notebooks/05_exploratory_data_analysis.ipynb
notebooks/06_tour_cost_analysis.ipynb

# Phase 3: Forecasting
notebooks/08_time_series_aggregation.ipynb       # Prepare time series (10 metrics)
notebooks/09_baseline_models.ipynb               # Baseline models
notebooks/10_prophet_model.ipynb                 # Prophet
notebooks/11_sarimax_model.ipynb                 # SARIMAX
notebooks/12_xgboost_model.ipynb                 # XGBoost
notebooks/14_consolidated_forecasts_2025.ipynb   # Select best models
notebooks/15_model_comparison.ipynb              # Compare performance
notebooks/17_monthly_forecast_2025_table.ipynb   # Generate forecast table

# Phase 4: Validation
notebooks/18_forecast_validation_2025.ipynb      # Validate vs actual 2025 data

# Phase 5: Reporting
python create_presentation.py                    # Generate PowerPoint
```

**Important**: After updating `utils/traveco_utils.py`, always **restart the Jupyter kernel**!

### Python Environment Setup

```bash
# Using workspace-level Pipenv
cd /Users/kk/dev
pipenv shell
cd customer_traveco
jupyter notebook

# Or install dependencies directly
pip install pandas numpy prophet scikit-learn xgboost statsmodels plotly tqdm scipy
```

## Key Notebook Details

### Notebook 08: Time Series Aggregation
- Prepares monthly time series data with **10 metrics**
- Calculates vehicle costs: (KM × KM Cost) + (Time × Minute Cost)
- Output: `data/processed/time_series_full_company.csv`

### Notebooks 09-12: Forecasting Models
- **09**: Baseline models (Seasonal Naive, Moving Average)
- **10**: Prophet with custom seasonalities (quarterly, monthly)
- **11**: SARIMAX with seasonal ARIMA components
- **12**: XGBoost with lag features and temporal features
- All support 10 metrics with backward compatibility

### Notebook 14: Consolidated Forecasts
- Selects best model per metric based on MAPE
- Generates final 2025 forecasts
- Output: `data/processed/consolidated_forecast_2025.csv`

### Notebook 18: Forecast Validation (⭐ KEY DELIVERABLE)
- Validates ML forecasts against actual 2025 data (Jan-Sep)
- Three-way comparison: Actual vs Human (2024÷12) vs Machine (ML)
- Generates 7 validation outputs:
  - `forecast_validation_error_comparison.html` - **CRUCIAL 4-panel dashboard**
  - `forecast_validation_orders_comparison.html`
  - `forecast_validation_revenue_comparison.html`
  - `forecast_validation_cumulative_error.html`
  - `forecast_validation_error_distribution.html`
  - `forecast_validation_summary.csv`
  - `forecast_validation_monthly_detail.csv`

## Data Dictionary (Quick Reference)

**Key Excel Files**:
1. `20251015 Juni 2025 QS Auftragsanalyse.xlsb` (23.6 MB) - Main order data
2. `20251015 QS Tourenaufstellung Juni 2025.xlsx` (2.85 MB) - Tour assignments
3. `20251015 Sparten.xlsx` (28 KB) - Customer divisions

**Critical Columns**:
- **Column G**: `Nummer.Auftraggeber` - Order owner (use for aggregation)
- **Column H**: `Id.Dispostelle` - Dispatch location (do NOT use for cost aggregation)
- **Column BC**: `Nummer.Spedition` - Carrier number (1-8889 = internal, 9000+ = external)
- **Column CU**: `Distanz_BE.Auftrag` - Distance (use this, not CV)
- **Column CY**: `Tilde.Auftrag` - Pickup flag (Ja = pickup, Nein = delivery)

**Data Filtering Rules**:
- Exclude: System B&T + empty customer (internal pickups)
- Exclude: "Lager Auftrag" (warehouse orders)
- Exclude: Orders with missing carrier numbers

**For complete data dictionary**: See `information/mail.pdf`

## Forecasting Model Configuration

### Prophet Parameters
```python
Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    seasonality_mode='multiplicative',
    changepoint_prior_scale=0.05
)
# Custom seasonalities
model.add_seasonality(name='quarterly', period=91.25, fourier_order=5)
model.add_seasonality(name='monthly', period=30.5, fourier_order=10)
```

### SARIMAX Parameters
```python
SARIMAX(
    ts_data,
    order=(2, 1, 2),              # ARIMA order
    seasonal_order=(1, 1, 1, 12), # Seasonal order
    enforce_stationarity=False,
    enforce_invertibility=False
)
```

### XGBoost Parameters
```python
xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)
```

## Validation Metrics

**Primary Metric**: MAPE (Mean Absolute Percentage Error)
- **Orders**: ML = 4.04%, Human = 4.28% → **ML wins by 5.6%**
- **Revenue**: Human = 5.39%, ML = 5.82% → **Human wins by 7.4%**

**Other Metrics**:
- **MAE** (Mean Absolute Error) - Absolute units
- **Seasonal MAPE** - Month-specific accuracy
- **Directional Accuracy** - Trend prediction correctness

## Current Project Status

- **Stage**: Production forecasting system with validation framework
- **Repository**: `git@github.com:kevinkuhn/customer_traveco.git`
- **Branch**: `feature/christian-feedback-corrections`
- **Data Coverage**: 2022-2024 historical + Jan-Nov 2025 actual data
- **Latest Enhancement**: Selective working_days feature based on correlation analysis
- **Key Achievement**: Complete ML validation framework
  - 11 months of 2025 actual data (Jan-Nov)
  - ML forecasts beat traditional by 5.6% for order volume
  - Traditional beats ML by 7.4% for revenue
  - **Recommendation**: Hybrid approach (ML for ops, traditional for revenue)

## Working with This Codebase

### To Modify Models

1. **Adjust seasonality**: Edit Prophet `add_seasonality()` calls in Notebook 10
2. **Change SARIMAX order**: Modify `order` and `seasonal_order` in Notebook 11
3. **Add lag features**: Update lag list in Notebook 12
4. **Change target metrics**: Update config in Notebook 08

### To Add New Validation Metrics

1. Update Notebook 18 Section 8 (metric calculation)
2. Add new visualization in Section 9
3. Update summary reports in Section 10

### Common Tasks

**Retrain models with new data**:
```bash
jupyter notebook notebooks/08_time_series_aggregation.ipynb  # Update time series
jupyter notebook notebooks/09_baseline_models.ipynb          # Retrain baseline
jupyter notebook notebooks/10_prophet_model.ipynb            # Retrain Prophet
jupyter notebook notebooks/11_sarimax_model.ipynb            # Retrain SARIMAX
jupyter notebook notebooks/12_xgboost_model.ipynb            # Retrain XGBoost
jupyter notebook notebooks/15_model_comparison.ipynb         # Compare
jupyter notebook notebooks/14_consolidated_forecasts_2025.ipynb  # Consolidate
```

**Validate new forecasts**:
```bash
jupyter notebook notebooks/18_forecast_validation_2025.ipynb
```

**Generate presentation**:
```bash
pipenv run python create_presentation.py
```

## Key Files and Utilities

### `utils/traveco_utils.py`
Core utility functions:
```python
class TravecomDataLoader:
    load_orders()           # Load main order data
    load_tours()            # Load tour assignments
    load_divisions()        # Load Sparten mapping

class TravecomFeatureEngine:
    extract_temporal_features()      # Year, month, week, etc.
    identify_carrier_type()          # Internal vs external
    map_customer_divisions()         # Sparten mapping (type-safe)
    map_betriebszentralen()          # Betriebszentralen mapping
    apply_filtering_rules()          # Exclude Lager orders
```

### `config/config.yaml`
Key settings:
```yaml
filtering:
  exclude_bt_pickups: true
  exclude_lager_orders: true

features:
  target_columns: ["total_orders", "total_km_billed", "total_km_actual",
                   "total_tours", "total_drivers", "revenue_total",
                   "external_drivers", "vehicle_km_cost",
                   "vehicle_time_cost", "total_vehicle_cost"]
```

## Domain Context

### Swiss Transport Industry Specifics
- **LSVA tax**: Heavy vehicle tax affecting costs
- **School vacations**: Demand seasonality driver
- **Cooled goods**: Summer peak seasonality
- **Regional economics**: Branch-specific demand patterns

### Data Scale
- **~1.2M records/year** in historical data
- **Monthly aggregation** for forecasting
- **14 Betriebszentralen** (dispatch centers)

## Documentation Files

- **CLAUDE.md**: This file - main project guide and development reference
- **README.md**: Project overview and quick start
- **TROUBLESHOOTING.md**: Common issues and solutions
- **FORECAST_METHODOLOGY.md**: Technical methodology documentation
- **results/FORECAST_MODEL_COMPARISON_FINAL.md**: Model comparison results
- **information/recommendation.md**: Strategic recommendations
- **information/mail.pdf**: Data field explanations from Traveco

## Next Steps

**For Production Deployment**:
1. Create `requirements.txt` with pinned versions
2. Add data validation and error handling
3. Implement unit tests for model methods
4. Create CLI or API wrapper (FastAPI) for forecast generation
5. Add logging and monitoring
6. Set up automated monthly retraining pipeline

**For Model Improvement**:
- Add more temporal features (Swiss holidays, weather, fuel prices)
- Test ensemble methods (weighted averaging of top models)
- Implement branch-level forecasting (per Betriebszentrale)
- Add confidence intervals to forecasts

**For Business Value**:
- Monthly forecast updates with actual vs predicted tracking
- Automated alerting for significant forecast deviations
- Integration with ERP/planning systems
- Executive dashboard for real-time monitoring
