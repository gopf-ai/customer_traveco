# Traveco Transport Forecasting Project

Advanced time series forecasting for Swiss transport logistics demand prediction (2025).

## Project Overview

This project implements a comprehensive multi-model forecasting system for **Traveco**, a Swiss transport/logistics company. The goal is to predict 2025 monthly transport metrics (revenue, external drivers, personnel costs) using historical order data.

### Client
**Traveco Transporte AG** - Swiss logistics company

### Objective
Forecast 2025 monthly metrics:
- Revenue
- External drivers (subcontractors)
- Personnel costs

### Approach
Dual model strategy combining:
- **Model A**: Seasonality-focused (Prophet, SARIMAX, XGBoost)
- **Model B**: Time decay-focused (Weighted Prophet, Time-weighted Ensemble)

## Quick Start

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt
```

Or use workspace-level Pipenv:

```bash
cd /Users/kk/dev
pipenv shell
cd customer_traveco
pip install -r requirements.txt
```

### 2. Launch Jupyter

```bash
jupyter lab
```

### 3. Run Notebooks in Sequence

Start with the first two notebooks:

1. **`notebooks/01_data_loading_and_exploration.ipynb`**
   - Load Traveco data files
   - Explore data structure
   - Understand column mappings

2. **`notebooks/02_data_cleaning_and_validation.ipynb`**
   - Clean and validate data
   - Handle missing values
   - Save clean dataset

## Project Structure

```
customer_traveco/
├── config/
│   └── config.yaml                 # Project configuration
├── data/
│   ├── swisstransfer_*/            # Raw data from Traveco (Excel files)
│   └── processed/                  # Cleaned and processed datasets
├── notebooks/
│   ├── 01_data_loading_and_exploration.ipynb
│   ├── 02_data_cleaning_and_validation.ipynb
│   └── ... (15 notebooks total - see CLAUDE.md)
├── utils/
│   └── traveco_utils.py            # Reusable Python utilities
├── models/                         # Saved model files
├── results/                        # Forecast outputs and reports
├── information/
│   ├── recommendation.md           # Strategic recommendations
│   └── mail.pdf                    # Data dictionary from client
├── CLAUDE.md                       # Complete development guide
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Data Files

Three main files provided by Traveco (in `data/swisstransfer_*/`):

1. **`20251015 Juni 2025 QS Auftragsanalyse.xlsb`** (23.6 MB)
   - Main order analysis data
   - ~1.2M records per year
   - Complete transport order history

2. **`20251015 QS Tourenaufstellung Juni 2025.xlsx`** (2.85 MB)
   - Tour assignments for June 2025
   - Links orders to specific tours

3. **`20251015 Sparten.xlsx`** (28 KB)
   - Customer division mappings
   - Product category classifications

## Key Features

### Data Processing
- Temporal feature extraction (year, month, quarter, week)
- Lag features (1, 3, 6, 12 months)
- Exponential time decay weighting
- Swiss business calendar integration

### Forecasting Models

**Model A: Seasonality-Focused**
- **Prophet**: Custom Swiss seasonalities (quarterly, monthly)
- **SARIMAX**: ARIMA(2,1,2) with seasonal (1,1,1,12)
- **XGBoost**: Non-linear temporal patterns

**Model B: Time Decay-Focused**
- **Weighted Prophet**: Exponential decay (λ=0.3-0.5)
- **Time-Weighted Ensemble**: Year-level RandomForest

### Evaluation Metrics
- MAPE (Mean Absolute Percentage Error)
- RMSE (Root Mean Square Error)
- Seasonal MAPE
- Directional Accuracy

## Configuration

All project settings are in `config/config.yaml`:

```yaml
data:
  raw_path: "data/swisstransfer_*/"
  processed_path: "data/processed/"

features:
  target_columns: ["revenue", "external_drivers", "personnel_costs"]
  lag_periods: [1, 3, 6, 12]

models:
  prophet:
    yearly_seasonality: true
    seasonality_mode: "multiplicative"
  # ... and more
```

## Utilities

The `utils/traveco_utils.py` module provides:

- **`ConfigLoader`**: Load YAML configuration
- **`TravecomDataLoader`**: Load Excel data files
- **`TravecomFeatureEngine`**: Feature engineering utilities
- **`TravecomDataCleaner`**: Data cleaning and validation
- Helper functions for metrics, time weighting, and data I/O

## Workflow

### Phase 1: Data Understanding (Notebooks 1-2)
1. Load and explore data
2. Clean and validate

### Phase 2: Feature Engineering (Notebooks 3-4)
3. Extract temporal features
4. Create aggregations

### Phase 3: Exploratory Analysis (Notebook 5)
5. Analyze patterns and seasonality

### Phase 4: Baseline Models (Notebook 6)
6. Establish performance benchmarks

### Phase 5-6: Advanced Modeling (Notebooks 7-11)
7-9. Model A implementations (Prophet, SARIMAX, XGBoost)
10-11. Model B implementations (Weighted Prophet, Ensemble)

### Phase 7: Model Selection (Notebooks 12-13)
12. Compare all models
13. Diagnostic analysis

### Phase 8: Production (Notebooks 14-15)
14. Generate final 2025 forecasts
15. Create dashboards and reports

## Documentation

- **`CLAUDE.md`**: Complete development guide with:
  - Detailed data dictionary
  - Model configurations
  - Swiss domain context
  - Notebook structure
  - Development commands

- **`information/recommendation.md`**: Strategic recommendations for model selection and implementation

- **`information/mail.pdf`**: Data field explanations from Traveco

## Development Status

**Current Stage**: Production forecasting system with validation framework
**Last Updated**: December 2025 (data through November 2025)

**Completed**:
- ✅ Complete data pipeline (cleaning, feature engineering, aggregation)
- ✅ 10 operational metrics forecasting
- ✅ Financial metrics forecasting (5 metrics)
- ✅ Prophet, SARIMAX, XGBoost model implementations
- ✅ Working days feature with correlation-based selection
- ✅ Validation against 11 months of 2025 actual data
- ✅ Interactive HTML dashboards

**Key Achievement**:
- ML forecasts beat traditional by 5.6% for order volume
- Traditional beats ML by 7.4% for revenue
- Recommendation: Hybrid approach (ML for operations, traditional for revenue)

## Support

For detailed guidance on working with this codebase, see **`CLAUDE.md`**.

For questions about the data structure, refer to **`information/mail.pdf`**.

---

**Author**: Kevin Kuhn
**Last Updated**: 2025-12-22
**Client**: Traveco Transporte AG
