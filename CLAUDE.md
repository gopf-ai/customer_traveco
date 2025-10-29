# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **transport logistics analysis project** for Traveco (Travecom), a Swiss transport/logistics company. The project performs operational analysis of June 2025 transport data with corrected business logic for accurate cost attribution and performance insights.

**Client**: Traveco (Switzerland)
**Objective**: Analyze transport operations with correct cost attribution, track order types (including Leergut), calculate vehicle costs, and measure KM efficiency
**Stage**: Operational analysis complete with corrected methodology; forecasting awaits historical data (24+ months needed)
**Key Achievement**: Corrected aggregation from dispatch location (Dispostelle) to order owner (Auftraggeber) for accurate cost attribution

## ⚠️ CRITICAL CORRECTIONS APPLIED (October 2025)

This project underwent a major methodology correction. If working with older code/notebooks, be aware:

### What Was Corrected:

1. **Cost Attribution** (MOST CRITICAL):
   - ❌ **OLD**: Aggregated by `Id.Dispostelle` (dispatch location) - Column H
   - ✅ **NEW**: Aggregated by `Nummer.Auftraggeber` (order owner/payer) - Column G
   - **Why**: Costs must be attributed to who pays for the transport, not who dispatches it
   - **Impact**: Notebook 04 cells 8-9 changed completely

2. **Order Type Classification**:
   - ❌ **OLD**: 2 categories (Pickup, Delivery)
   - ✅ **NEW**: 5 categories (Delivery, Pickup/Multi-leg, Leergut/Empty Returns, Retoure, Abholung)
   - **Why**: Leergut (empty container returns) are ~18% of orders and need separate tracking
   - **Impact**: Notebook 03 cell 11, Notebook 04 cell 9

3. **Data Filtering**:
   - ❌ **OLD**: Included all orders
   - ✅ **NEW**: Exclude "Lager Auftrag" (warehouse orders) and orders with missing carrier numbers
   - **Why**: Warehouse operations are not transport operations
   - **Impact**: `utils/traveco_utils.py` `apply_filtering_rules()`, `config.yaml`

4. **Sparten Mapping**:
   - ❌ **OLD**: Type mismatch caused zero matches (string vs int customer numbers)
   - ✅ **NEW**: Auto-convert both to Int64 for reliable matching
   - **Why**: Orders had string customer IDs, Divisions had integers
   - **Impact**: `utils/traveco_utils.py` `map_customer_divisions()`, Notebook 03 cell 16

5. **Betriebszentralen Mapping** (October 2025):
   - ❌ **OLD**: Aggregated by numeric Auftraggeber IDs (3000, 5000, etc.) - 12 entities
   - ✅ **NEW**: Mapped to Betriebszentralen names (BZ Oberbipp, LC Nebikon, etc.) - 14 entities
   - **Why**: The 14 Betriebszentralen are the actual invoicing units (dispatch centers)
   - **Impact**: Added `data/raw/TRAVECO_Betriebszentralen.csv`, new column `betriebszentrale_name`
   - **Files**: Notebook 03 Section 7.5, Notebook 04 cells 8-9, Notebooks 05-06, `create_presentation.py`

6. **New Analyses Added**:
   - ✅ Tour cost calculation (KM + time components) - Notebook 06
   - ✅ KM efficiency analysis (actual vs billed) - Notebook 05 Section 12
   - ✅ Sparten distribution analysis - Notebook 05 Section 13
   - ✅ Updated presentation with corrected insights

7. **Forecasting Metrics Expansion** (October 2025):
   - ❌ **OLD**: 5 target metrics (total_orders, total_km, total_drivers, revenue_total, external_drivers)
   - ✅ **NEW**: 10 target metrics (added tour-based metrics + vehicle cost metrics)
   - **Why**: Enable complete financial forecasting (revenue vs cost) and profit margin predictions
   - **Changes**:
     - `total_km` → `total_km_billed` (order-based billing KM for revenue forecasts)
     - **NEW** (7 metrics): `total_km_actual` (tour-based actual driven KM)
     - **NEW** (7 metrics): `total_tours` (unique tour count for resource planning)
     - **NEW** (10 metrics): `vehicle_km_cost` (KM-based transportation cost component)
     - **NEW** (10 metrics): `vehicle_time_cost` (Time-based transportation cost component)
     - **NEW** (10 metrics): `total_vehicle_cost` (Total vehicle operational cost)
   - **Cost Formula**: `Total Cost = (Actual KM × PC KM Kosten) + (IST Zeit PraCar × 60 × PC Minuten Kosten)`
   - **Impact**: Notebooks 08-12 updated to support 10 metrics with backward compatibility
   - **Files**:
     - `notebooks/08_time_series_aggregation.ipynb` - Cell-6 (cost calculation), Cell-16 (aggregation), Cell-18 (5-panel visualization)
     - All forecasting notebooks (09-12) - Updated target_metrics lists
   - **New Capabilities**: Revenue forecasting, Cost forecasting, Profit margin forecasting

### Files Most Affected:
- `notebooks/04_aggregation_and_targets.ipynb` - **Complete rewrite of aggregation logic**
- `notebooks/03_feature_engineering.ipynb` - Enhanced order types, fixed Sparten, added Betriebszentralen mapping
- `notebooks/05_exploratory_data_analysis.ipynb` - Added Sections 12-13, updated branch detection
- `notebooks/06_tour_cost_analysis.ipynb` - **NEW notebook**, uses Betriebszentralen
- `utils/traveco_utils.py` - Added `map_betriebszentralen()`, `load_betriebszentralen()`
- `create_presentation.py` - Updated with all corrections (13 slides), uses Betriebszentralen names
- `data/raw/TRAVECO_Betriebszentralen.csv` - **NEW file** with 14 dispatch center mappings

### If You See Errors:
- **"AttributeError: map_customer_divisions"**: Restart Jupyter kernel (see `TROUBLESHOOTING.md`)
- **"AttributeError: map_betriebszentralen"**: Restart Jupyter kernel after updating `traveco_utils.py`
- **"betriebszentrale_name not found"**: Run Notebook 03 with updated Betriebszentralen mapping section
- **All orders show "Keine Sparte"**: Sparten mapping type mismatch - fixed in latest code
- **Only 12 entities instead of 14**: Missing Betriebszentralen mapping - rerun Notebook 03

## Technology Stack

- **Language**: Python
- **Primary Environment**: Jupyter Notebooks
- **Key Libraries**:
  - **Analysis**: pandas, numpy, scipy
  - **Visualization**: Plotly (interactive dashboards), matplotlib, seaborn
  - **Presentation**: python-pptx (PowerPoint generation)
  - **Data**: openpyxl (Excel), pyxlsb (.xlsb files)
  - **Forecasting** (future): Prophet, SARIMAX, XGBoost (awaiting historical data)

## Repository Structure

```
customer_traveco/
├── notebooks/
│   ├── 02_data_cleaning_and_validation.ipynb    # Data cleaning with corrected filtering
│   ├── 03_feature_engineering.ipynb              # Features + Sparten mapping (fixed type mismatch)
│   ├── 04_aggregation_and_targets.ipynb          # Aggregation by Auftraggeber (CORRECTED)
│   ├── 05_exploratory_data_analysis.ipynb        # EDA + KM efficiency + Sparten charts
│   └── 06_tour_cost_analysis.ipynb               # Vehicle cost calculation + efficiency
├── data/
│   ├── raw/
│   │   └── TRAVECO_Betriebszentralen.csv         # 14 dispatch centers mapping (NEW)
│   ├── swisstransfer_*/         # Raw Excel files with transport order data
│   │   ├── 20251015 Sparten.xlsx                    # Customer divisions (Sparten)
│   │   ├── 20251015 QS Tourenaufstellung Juni 2025.xlsx  # Tour assignments
│   │   └── 20251015 Juni 2025 QS Auftragsanalyse.xlsb   # Order analysis (23.6 MB)
│   └── processed/               # Processed data files
│       ├── clean_orders.csv                      # Cleaned orders (Lager excluded)
│       ├── features_engineered.csv               # With Leergut, Sparten, Betriebszentralen, temporal features
│       ├── monthly_aggregated.csv                # By Betriebszentralen (14 dispatch centers)
│       ├── tour_costs.csv                        # Vehicle costs + KM efficiency
│       └── orders_with_tour_costs.csv            # Orders joined with tour-level costs
├── utils/
│   └── traveco_utils.py         # Core utilities (enhanced with Sparten mapping)
├── results/                      # Generated charts and dashboards
│   ├── km_efficiency_dashboard.html              # Interactive efficiency analysis
│   ├── sparten_distribution_dashboard.html       # Customer division insights
│   └── Traveco_Data_Insights_June2025_Corrected.pptx  # Updated presentation
├── config/
│   └── config.yaml              # Configuration (exclude_lager_orders: true)
├── create_presentation.py       # PowerPoint generator (UPDATED with Betriebszentralen)
├── debug_sparten_mapping.py     # Debug script for Sparten type mismatch
├── BETRIEBSZENTRALEN_MIGRATION.md  # Technical documentation for Betriebszentralen mapping
├── TROUBLESHOOTING.md           # Module caching and common issues
└── information/
    ├── recommendation.md        # Strategic forecasting recommendations
    └── mail.pdf                 # Client communication
```

## Development Commands

### Running the Forecasting Notebooks

This is a Jupyter-based project. To work with the code:

```bash
# Start Jupyter from project root
jupyter notebook

# Or use Jupyter Lab
jupyter lab

# Navigate to notebooks/
```

### Recommended Notebook Structure

For building this project step-by-step, create the following notebook sequence:

#### Phase 1: Data Understanding

**`01_data_loading_and_exploration.ipynb`**
- Load three Excel files (Auftragsanalyse, Tourenaufstellung, Sparten)
- Initial data exploration (shape, dtypes, head/tail, info)
- Understand column mappings from data dictionary
- Check for missing values and data completeness
- Apply filtering rules (exclude B&T pickup orders)
- **Output**: Raw data loaded and understood

**`02_data_cleaning_and_validation.ipynb`**
- Handle missing values (decide on strategy: drop, fill, interpolate)
- Validate business rules from data dictionary
- Data type conversions (dates, numeric fields)
- Outlier detection and handling
- Create clean dataset with documentation of changes
- **Output**: `data/processed/clean_orders.csv`

#### Phase 2: Feature Engineering

**`03_feature_engineering.ipynb`**
- Extract temporal features (year, month, week, quarter, day_of_year, weekday)
- Calculate distances (use Column CU: Distanz_BE, not CV)
- Identify pickup vs delivery (Tilde column)
- Map customers to divisions (Sparten)
- Add carrier type flag (internal vs external based on carrier number)
- Calculate time decay weights
- Create lag features (1, 3, 6, 12 months)
- **Output**: `data/processed/features_engineered.csv`

**`04_aggregation_and_targets.ipynb`** ⚠️ CRITICAL CORRECTION APPLIED
- **CORRECTED**: Monthly aggregation by **Auftraggeber (Column G)** instead of Dispostelle
- This ensures costs are attributed to order owner/payer, not dispatch location
- Enhanced order type breakdown: delivery, pickup, leergut, retoure
- Calculate target variables: total_orders, distances, external/internal drivers
- Dual-view aggregation for comparison (both Auftraggeber and Dispostelle)
- **Output**: `data/processed/monthly_aggregated.csv` (by Auftraggeber)
- **Location**: Cells 8-9 contain the critical aggregation logic

#### Phase 3: Exploratory Analysis

**`05_exploratory_data_analysis.ipynb`** ✅ ENHANCED WITH NEW SECTIONS
- Time series visualization (trends, seasonality)
- Seasonal decomposition (trend, seasonal, residual)
- Entity-level analysis (Auftraggeber or Dispostelle)
- Correlation analysis between features
- **Section 12** (NEW): Kilometer Efficiency Analysis
  - 4-panel dashboard: distribution, categories, actual vs billed, efficiency by size
  - Identifies tours with >10% inefficiency for optimization
  - Output: `results/km_efficiency_dashboard.html`
- **Section 13** (NEW): Customer Division (Sparten) Analysis
  - Top 15 Sparten by volume, distribution pie chart
  - Distance and carrier patterns by Sparten
  - Output: `results/sparten_distribution_dashboard.html`
- **Output**: Insights document, interactive dashboards, key visualizations saved

#### Phase 4: Tour Cost Analysis (Replaces Baseline Models for now)

**`06_tour_cost_analysis.ipynb`** ✅ NEW NOTEBOOK
- Calculate full vehicle costs: (KM × KM Cost) + (Time × 60 × Minute Cost)
- Load tour data from `20251015 QS Tourenaufstellung Juni 2025.xlsx`
- Join PraCar system data (actual km, actual time, costs)
- Calculate KM efficiency ratio: Actual KM / Billed KM
- Categorize efficiency: Excellent (<0.9), Good (0.9-1.0), Acceptable (1.0-1.1), Poor (>1.1)
- **Output**: `data/processed/tour_costs.csv`, `data/processed/orders_with_tour_costs.csv`
- **Note**: Forecasting models deferred until historical data (24+ months) is available

#### Phase 5: Reporting and Presentation

**`create_presentation.py`** ✅ UPDATED
- Generates PowerPoint presentation with corrected analysis
- Dynamically adapts to Auftraggeber vs Dispostelle aggregation
- Includes new slides:
  - Slide 7: KM Efficiency Analysis (NEW)
  - Slide 8: Sparten (Customer Division) Analysis (NEW)
  - Updated slides with corrected insights and recommendations
- **Output**: `results/Traveco_Data_Insights_June2025_Corrected.pptx` (13 slides)
- **Run**: `pipenv run python create_presentation.py`

---

## 🔮 FUTURE WORK: Time Series Forecasting (Awaiting Historical Data)

**Status**: Waiting for historical data (~36 months back from June 2025)
**Required**: Data from ~June 2022 to June 2025 for robust forecasting
**Objective**: Predict 2025+ monthly transport metrics (total_orders, total_km, total_drivers, revenue_total, external_drivers)

Once historical data is available, implement the following forecasting workflow:

### Phase 6: Baseline Forecasting Models

**`07_baseline_models.ipynb`** (TO BE CREATED)
- Current method: Linear averaging (yearly average / 12)
- Simple moving average (3-month, 6-month, 12-month)
- Naive forecast (last year same month)
- Seasonal naive (repeat last year's pattern)
- Calculate baseline metrics (MAPE, RMSE)
- **Purpose**: Establish performance benchmarks to beat
- **Output**: `results/baseline_forecasts.csv`

### Phase 7: Advanced Models - Seasonality Focus (Model Family A)

**`08_model_a_prophet.ipynb`** (TO BE CREATED)
- Prophet implementation with custom seasonalities
- Hyperparameter tuning (changepoint_prior_scale, seasonality parameters)
- Add Swiss holidays (National Day, Christmas, Easter, etc.)
- Test custom seasonalities:
  - Quarterly (91.25 days) - for seasonal transport patterns
  - Monthly (30.5 days) - for monthly business cycles
- Cross-validation using TimeSeriesSplit (5-fold recommended)
- Generate 2025/2026 forecasts with confidence intervals
- **Output**: `models/prophet_model.pkl`, forecasts, performance metrics

**`09_model_a_sarimax.ipynb`** (TO BE CREATED)
- SARIMAX (Seasonal ARIMA) implementation
- Order selection process:
  - Use ACF/PACF plots from notebook 05 to determine (p, d, q)
  - Test seasonal orders: (P, D, Q, s) with s=12 for monthly data
  - Grid search over reasonable ranges (e.g., p,q ∈ [0,3], P,Q ∈ [0,2])
- Diagnostic plots (residuals, ACF, PACF, Q-Q plot)
- Stationarity tests (Augmented Dickey-Fuller test)
- Ljung-Box test for residual autocorrelation
- Generate 2025/2026 forecasts
- **Output**: `models/sarimax_model.pkl`, forecasts, diagnostics

**`10_model_a_xgboost.ipynb`** (TO BE CREATED)
- XGBoost with temporal and engineered features
- Features to include:
  - Temporal: year, month, week, quarter, day_of_year, weekday
  - Lag features: [1, 3, 6, 12] months
  - Rolling statistics: 3-month, 6-month averages
  - Sparten-level aggregations
  - External/internal driver ratios
  - KM efficiency metrics
- Hyperparameter tuning via GridSearchCV or RandomizedSearchCV:
  - n_estimators: [100, 200, 300]
  - max_depth: [3, 6, 9]
  - learning_rate: [0.01, 0.05, 0.1]
  - subsample: [0.7, 0.8, 0.9]
- Feature importance analysis (identify key drivers)
- Cross-validation (TimeSeriesSplit)
- Generate 2025/2026 forecasts
- **Output**: `models/xgboost_model.pkl`, forecasts, feature importance plot

### Phase 8: Advanced Models - Time Decay Focus (Model Family B)

**Rationale**: Recent data may be more relevant due to market changes, new routes, customer shifts

**`11_model_b_weighted_prophet.ipynb`** (TO BE CREATED)
- Prophet with exponential time decay weighting
- Weight formula: `w(t) = exp(-λ * (T_current - t) / 365)`
- Test different decay rates (λ):
  - λ = 0.2 (slow decay, ~5-year half-life)
  - λ = 0.3 (medium decay, ~3-year half-life)
  - λ = 0.5 (fast decay, ~2-year half-life)
- Implementation approaches:
  - Data duplication: Replicate recent observations proportional to weights
  - Custom Prophet weights (if supported in newer versions)
- More responsive changepoint detection (changepoint_prior_scale=0.1)
- Compare against standard Prophet from notebook 08
- Generate 2025/2026 forecasts
- **Output**: `models/weighted_prophet_model.pkl`, forecasts, decay rate comparison

**`12_model_b_time_weighted_ensemble.ipynb`** (TO BE CREATED)
- Year-level RandomForest ensemble with weighted voting
- Strategy:
  - Train separate RandomForest models for each year in historical data
  - Assign exponential weights to year-level models: `exp(-0.5 * (2025 - year))`
  - Combine predictions using weighted voting
- Alternative: Train single model with year-specific features + weighted loss
- Test different ensemble strategies:
  - Simple averaging vs weighted averaging
  - Include/exclude oldest years (sensitivity analysis)
- Generate 2025/2026 forecasts
- **Output**: `models/time_weighted_ensemble.pkl`, forecasts, year weight analysis

### Phase 9: Model Comparison and Selection

**`13_model_comparison.ipynb`** (TO BE CREATED)
- Load all model forecasts from notebooks 07-12
- Calculate comprehensive performance metrics:
  - **MAPE** (Mean Absolute Percentage Error) - Primary metric
  - **RMSE** (Root Mean Square Error) - Penalizes large errors
  - **MAE** (Mean Absolute Error) - Interpretable in original units
  - **Seasonal MAPE** - By quarter and by month
  - **Directional accuracy** - % of correct trend predictions
  - **Forecast bias** - Systematic over/under-prediction
- Statistical significance tests:
  - Diebold-Mariano test (compare forecast accuracy)
  - Paired t-tests for MAPE differences
- Model ranking by metric (with confidence intervals)
- Analyze model strengths/weaknesses:
  - Which models perform best by season?
  - Which models perform best by branch/Sparte?
  - Which models handle outliers better?
- Create ensemble of best models:
  - Weighted average based on validation performance
  - Test: Equal weights vs optimized weights (minimize validation MAPE)
- **Output**: `results/model_comparison_report.csv`, `results/model_rankings.png`

**`14_model_diagnostics.ipynb`** (TO BE CREATED)
- Deep dive into model residuals and error patterns
- Residual analysis for each model:
  - Distribution (normality check)
  - Autocorrelation (Ljung-Box test)
  - Heteroscedasticity (constant variance check)
  - Q-Q plots
- Forecast error patterns:
  - Errors by time period (recent vs older data)
  - Errors by seasonality (summer vs winter)
  - Errors by magnitude (small vs large orders)
- Branch/Entity-specific performance:
  - Which Auftraggeber are hardest to predict?
  - Are there systematic biases by branch?
- Sparten-specific performance:
  - Which customer divisions are most predictable?
  - Do certain Sparten have different seasonal patterns?
- Identify failure modes:
  - When do models fail? (Holidays, outlier months, structural changes)
  - Are there systematic patterns in errors?
- **Output**: `results/model_diagnostics_report.html`, improvement recommendations

### Phase 10: Production Forecasts

**`15_final_forecast_2025_2026.ipynb`** (TO BE CREATED)
- Retrain best-performing model(s) on full dataset (no holdout)
- Generate final production forecasts:
  - Monthly forecasts for remaining 2025 months
  - Monthly forecasts for full 2026
  - Optionally: 2027 (with wider confidence intervals)
- Create confidence intervals:
  - 80% prediction intervals (typical business use)
  - 95% prediction intervals (conservative planning)
- Branch/Entity-level forecasts:
  - Forecast for each Auftraggeber independently
  - Aggregate to company-level
- Division-level forecasts:
  - Forecast by Sparten (customer division)
  - Identify high-growth vs declining segments
- Scenario analysis:
  - Best case (95th percentile)
  - Base case (median)
  - Worst case (5th percentile)
- **Output**:
  - `results/forecast_2025_2026.csv`
  - `results/forecast_by_auftraggeber.csv`
  - `results/forecast_by_sparte.csv`

**`16_visualization_and_reporting.ipynb`** (TO BE CREATED)
- Create comprehensive interactive Plotly dashboards
- Visualizations:
  - Historical vs forecast comparison (with confidence bands)
  - Model A vs Model B vs Ensemble comparison
  - Current method vs new approach (improvement quantification)
  - Branch-level dashboards (drill-down by Auftraggeber)
  - Sparten-level dashboards (customer division insights)
  - Seasonal pattern visualization
  - Growth rate analysis (YoY, MoM)
- Export charts for presentation (PNG, PDF)
- Create executive summary report:
  - Key findings
  - Forecast highlights
  - Risk factors and uncertainties
  - Recommended actions
- **Output**:
  - `results/forecast_dashboard.html` (interactive)
  - `results/forecast_presentation.pptx` (executive summary)
  - Chart exports: `results/charts/*.png`

### Phase 11: Deployment and Monitoring (Optional)

**`17_forecast_automation.py`** (TO BE CREATED)
- Automate monthly forecast updates
- Pipeline:
  1. Load new month's data
  2. Update feature engineering
  3. Retrain model (or use pre-trained)
  4. Generate next 12-month forecasts
  5. Compare actuals vs previous forecasts (track accuracy)
  6. Generate alert emails if significant deviations detected
- Schedule via cron job or cloud scheduler
- **Output**: Automated forecast pipeline

**`18_forecast_monitoring.ipynb`** (TO BE CREATED)
- Track forecast accuracy over time
- Monthly actuals vs forecast comparison
- Rolling MAPE calculation
- Detect model drift (when to retrain)
- Alert system for significant forecast misses
- **Output**: `results/forecast_accuracy_dashboard.html`

---

### Key Success Factors for Forecasting

1. **Data Quality**: Ensure 36 months of clean, consistent data
2. **Seasonality**: Swiss-specific patterns (holidays, school vacations, weather)
3. **Business Logic**: Continue using corrected Auftraggeber aggregation
4. **Feature Engineering**: Leverage Sparten, carrier types, Leergut tracking
5. **Validation**: Use proper time series cross-validation (no data leakage)
6. **Interpretability**: Prefer models that business stakeholders can understand (Prophet > XGBoost for presentations)
7. **Ensemble**: Combine multiple models for robustness

### Recommended Timeline (Once Data Received)

- **Weeks 1-2**: Data integration, validation, update notebooks 02-04
- **Weeks 3-4**: Baseline and Model A implementations (notebooks 07-10)
- **Weeks 5-6**: Model B implementations (notebooks 11-12)
- **Week 7**: Model comparison and selection (notebooks 13-14)
- **Week 8**: Production forecasts and reporting (notebooks 15-16)
- **Total**: ~8 weeks to production forecasts

See `information/recommendation.md` for additional strategic guidance.

### Supporting Files and Utilities

**`utils/traveco_utils.py`** ✅ UPDATED (Python module)
Key classes and methods:
```python
class TravecomDataLoader:
    """Load and validate Traveco data files"""
    load_orders()           # Load main order data (.xlsb)
    load_tours()            # Load tour assignments
    load_divisions()        # Load Sparten mapping

class TravecomFeatureEngine:
    """Feature engineering utilities"""
    extract_temporal_features()      # Year, month, week, quarter, etc.
    identify_carrier_type()          # Internal vs external
    calculate_time_weights()         # Exponential time decay
    map_customer_divisions()         # ✅ FIXED: Auto-converts types for matching
    apply_filtering_rules()          # ✅ UPDATED: Excludes Lager orders
```

**`config/config.yaml`** ✅ UPDATED (Configuration file)
Key settings:
```yaml
data:
  raw_path: "data/swisstransfer_*/"
  processed_path: "data/processed/"
  clean_orders: "clean_orders.csv"
  features_engineered: "features_engineered.csv"
  monthly_aggregated: "monthly_aggregated.csv"

filtering:
  exclude_bt_pickups: true        # Exclude B&T internal pickups
  exclude_lager_orders: true      # ✅ NEW: Exclude warehouse orders

features:
  target_columns: ["total_orders", "total_km_billed", "total_km_actual", "total_tours", "total_drivers", "revenue_total", "external_drivers"]

time_decay:
  decay_rate: 0.3                 # λ for exponential decay
```

**`debug_sparten_mapping.py`** (Diagnostic script)
- Debugs Sparten mapping type mismatches
- Checks customer number data types in both files
- Identifies matching issues
- Run: `python debug_sparten_mapping.py`

**`TROUBLESHOOTING.md`** (Common issues guide)
- Module caching (AttributeError for new methods)
- Sparten mapping failures
- Kernel restart procedures
- File path issues

### Current Notebook Execution Order

Execute notebooks in this sequence:

```bash
# Phase 1-2: Data Preparation
jupyter notebook notebooks/02_data_cleaning_and_validation.ipynb
jupyter notebook notebooks/03_feature_engineering.ipynb
jupyter notebook notebooks/04_aggregation_and_targets.ipynb

# Phase 3: Analysis
jupyter notebook notebooks/05_exploratory_data_analysis.ipynb
jupyter notebook notebooks/06_tour_cost_analysis.ipynb

# Phase 4: Reporting
pipenv run python create_presentation.py
```

**Important**: After updating `utils/traveco_utils.py`, always **restart the Jupyter kernel** before re-running notebooks!

### Python Environment Setup

No `requirements.txt` exists yet. Install dependencies manually:

```bash
pip install pandas numpy prophet scikit-learn xgboost statsmodels plotly tqdm scipy
```

Or if using the workspace-level Pipenv (from parent `/Users/kk/dev/`):

```bash
cd /Users/kk/dev
pipenv shell
cd customer_traveco
jupyter notebook
```

### Working with the Data

**Data Loading Pattern** (expected in notebooks):
```python
import pandas as pd

# Load order analysis data
data = pd.read_excel('data/swisstransfer_*/20251015 Juni 2025 QS Auftragsanalyse.xlsb')

# Expected schema:
# - tour_start_date (datetime)
# - auftragsschein_id (order ID)
# - branch (branch identifier)
# - revenue (numeric)
# - external_drivers (count)
# - personnel_costs (numeric)
```

## Data Dictionary

The following detailed field explanations are provided by Traveco (see `information/mail.pdf`). These define the structure of files in `data/swisstransfer_*/`:

### 1. Data Filtering Rules

**Exclude B&T Pickup Orders** (not relevant for analysis):
- Filter: System B&T (Column CW) = "B&T" AND Customer (RKDNr Column L) = empty
- These are internal pickup orders that should not be included in forecasting

### 2. Transport Type Classifications

**Column Mappings**:
- **Column K**: `Auftragsart` (Order type)
- **Column AU**: `Lieferart 2.0` (Delivery type)
- **Column CW**: `System_id.Auftrag` (System ID)
- **Column AR**: `Text für Auswertung` (Description)
- **Column AS**: `∑ Einheiten` (Total units)
- **Column AT**: `∑ Frachtpfl. Gewicht` (Freight weight)

**Transport Categories**:

| Order Type | Delivery Type | System | Units | Weight | Description |
|------------|---------------|--------|-------|--------|-------------|
| - | **B&T Fossil** | B&T | x100 ist lt | - | Fossil fuel delivery |
| - | **B&T Holzpellets** | B&T | x100 ist kg | - | Wood pellets delivery |
| - | **Flüssigtransporte** | TRP | - | ist kg | Liquid transport |
| - | **Lager Auftrag** | TRP | - | - | Warehouse orders (not relevant) |
| - | **Losetransporte** | TRP | - | ist kg | Bulk transport (do not use) |
| Leergut | **Palettentransporte** | TRP | EH alle TM in SBB Kg | kg | Empty pallet returns |
| Lieferung | **Palettentransporte** | TRP | EH alle TM in SBB Kg | kg | Pallet delivery |
| Retoure/Abholung | **Palettentransporte** | TRP | EH alle TM in SBB Kg | kg | Pallet return/pickup |

### 3. Carrier Identification (Spediteure)

**Columns**:
- **Column BC**: `Nummer.Spedition` (Carrier number)
- **Column BD**: `Name.Spedition` (Carrier name)

**Carrier Code Ranges**:
- **1 - 8889**: TRAVECO internal carriers
- **9000+**: External carriers (Fremdfahrer)

**TRAVECO Internal Carriers**:

| Code | Name |
|------|------|
| 1100 | B&T Winterthur |
| 1200 | B&T Landquart |
| 1500 | BZ Intermodal / Rail |
| 1600 | B&T Puidoux |
| 1900 | BZ Sierre |
| 3000 | BZ Oberbipp |
| 4000 | BZ Winterthur |
| 5000 | BZ Nebikon |
| 6000 | BZ Herzogenbuchsee |
| 6040-6042 | BZ/B&T Puidoux |
| 7000 | BZ Landquart |

**External Carriers (examples)**:

| Code | Name |
|------|------|
| 9000 | Benz Steffen Transporte |
| 9004 | Bachmann AG Transporte Schweiz |
| 9013 | Blättler Transport AG |

### 4. Pickup vs Delivery Identification

**Column CY**: `Tilde.Auftrag` (Tilde flag)
- **"Ja" (Yes)**: Pickup order (Vorholung) - marked with `~` in order number
- **"Nein" (No)**: Delivery order (Auslieferung)

**Distance Calculation**:
- **Use Column CU** (`Distanz_BE.Auftrag`): Distance from loading location to unloading location
- **Do NOT use Column CV** (`Distanz_VE.Auftrag`): Sender to receiver distance

**Cross-Docking Example**:
When an order routes through a cross-dock facility (e.g., Sursee), the order is split:
- Pickup leg: Order number ends with `~001` (Tilde = "Ja")
- Delivery leg: Same order number without tilde (Tilde = "Nein")

Example from data:
```
Hauptauftrag: RAM0081584480_0
  - Pickup:   RAM0081584480~001 | Date: 02.06.2025 | Tour: 565058 | HOCHDORF → Sursee
  - Delivery: RAM0081584480     | Date: 03.06.2025 | Tour: 567567 | Sursee → GEBENSTORF
```

### 5. Customer Billing Information

**Columns**:
- **Column L**: `RKdNr` (Customer billing number)
- **Column M**: `RKdArt` (Customer type: R = regular)
- **Column N**: `RKdName` (Customer name)
- **Column O**: `RKdOrt` (Customer city)

**Product Divisions (Sparten)**:
Customers are categorized into divisions. See separate file: `20251015 Sparten.xlsx` for mapping of customer numbers to divisions.

Example:
```
RKdNr: 946200 | Type: R | Name: Volg Konsumwaren AG | City: Winterthur
```

### 6. Loading and Unloading Locations

**Loading Location (Beladestelle)**:
- **Column AD**: `Nummer.Beladestelle` (Loading location ID)
- **Column AE**: `Name.Beladestelle` (Loading location name)
- **Column AF**: `Beladestelle Name 2` (Alternative name)
- **Column AG**: `Land.Beladestelle` (Country code: CH)
- **Column AH**: `Strasse.Beladestelle` (Street)
- **Column AI**: `PLZ.Beladestelle` (Postal code)
- **Column AJ**: `Ort.Beladestelle` (City)

**Unloading Location (Entladestelle)**:
- **Column AK**: `Nummer.Entladestelle` (Unloading location ID)
- Followed by similar fields (Name, Address, etc.)

Example:
```
Loading:   0020 | RAMSEIER Suisse AG | CH | Niffel | 6280 | HOCHDORF
Unloading: 5000 | BZ Sursee | TRAVECO Transporte AG | CH | Obstfeldstrasse 7 | 6210 | Sursee
```

### 7. Sender and Receiver Information

**Sender (Versender)**:
- **Column P**: `Nummer.Versender` (Sender ID)
- **Column Q**: `Name.Versender` (Sender name)
- **Column R**: `Versender Name 2` (Alternative name)
- **Column S**: `Strasse.Versender` (Street)
- **Column T**: `Land.Versender` (Country)
- **Column U**: `Ort.Versender` (City)
- **Column V**: `PLZ.Versender` (Postal code)

**Receiver (Empfänger)**:
- **Column W**: `Nummer.Empfänger` (Receiver ID)
- **Column X**: `Name.Empfänger` (Receiver name)
- Plus corresponding address fields

Example:
```
Sender:   0020 | RAMSEIER Suisse AG | Niffel | CH | HOCHDORF | 6280
Receiver: RAM0000205639 | LANDI Laden Gebenstorf | Hornblick 3 | CH | GEBENSTORF | 5412
```

### 8. Dispatch Locations (Dispostellen)

**Column H**: `Id.Dispostelle` (Dispatch location ID)

**TRAVECO Dispatch Centers**:

| ID | Name |
|----|------|
| 1_TRP_Lahr | Lahr dispatch center |
| 4_TRP_Winterthur | Winterthur dispatch |
| 5_TRP_Sursee | Sursee dispatch |
| 6_TRP_Herzogenbuchsee (EP) | Herzogenbuchsee (endpoint) |
| 7_TRP_Landquart | Landquart dispatch |
| 14_TRP_Oberbipp | Oberbipp dispatch |
| 15_TRP_Intermodal | Intermodal rail |
| 16_TRP_Herzogenbuchsee (DP) | Herzogenbuchsee (dispatch point) |
| 18_TRP_Puidoux (DP) | Puidoux dispatch |
| 19_TRP_Sierre | Sierre dispatch |
| 1000_TRP_Lager Nebikon | Nebikon warehouse |
| 90_TRP_Lager Hägendorf | Hägendorf warehouse |
| 99_TRP_Fakturierung | Billing/invoicing |

**B&T Dispatch Centers**:

| ID | Name |
|----|------|
| 1_B&T_1100 Ohringen | Ohringen B&T |
| 2_B&T_1200 Landquart | Landquart B&T |
| 6_B&T_1600 Puidoux | Puidoux B&T |
| 7_B&T_1150 OH-PELO | OH-PELO B&T |

### 9. Tour Information

**Columns**:
- **Column D**: `Datum.Tour` (Tour date)
- **Column E**: `Nummer.Tour` (Tour number)
- **Column F**: `Tour Bezeichnung` (Tour description)
- **Column CT**: `TourTyp.Tour` (Tour type code)

**Tour Type Codes**:

| Code | Description |
|------|-------------|
| **FD** | Fahrdienstleistung (Driving service) |
| **SH** | Subunternehmer (Subcontractor) |
| **Normal** | Standard tour |
| **TS** | Tour special |
| **GE** | Gebiet (Regional) |
| **Direktlieferung** | Direct delivery |
| **Bahn Vor-/Nachlauf** | Rail pre/post-haul |
| **Intermodal Bahn beladen** | Intermodal rail loaded |
| **Intermodal Bahn leer** | Intermodal rail empty |

Example:
```
Date: 01.06.2025 | Tour: 557265 | Name: 2130 Kuoni Samedan | Type: FD
```

### 10. PraCar System Data (Tour Costs)

**Columns**:
- **Column A**: Tour date (duplicate of Column D)
- **Column B**: Tour ID
- **Column K**: `Fahrzeug KM` (Vehicle kilometers from PraCar system)
- **Column L**: `Minuten Kosten` (Minute costs from PraCar)
- **Column T**: `Soll KM der Tour` (Target/planned kilometers)
- **Column V**: `IST Zeit PraCar` (Actual time from PraCar)
- **Column AB**: `Tour End (errechnet)` (Calculated tour end time)

**Note**: PraCar is Traveco's internal fleet management system that tracks actual vehicle performance and costs.

### Data Files Summary

**Three main files provided by Traveco**:

1. **`20251015 Juni 2025 QS Auftragsanalyse.xlsb`** (23.6 MB)
   - Main order analysis data
   - Contains all columns described above
   - Full historical order data for forecasting

2. **`20251015 QS Tourenaufstellung Juni 2025.xlsx`** (2.85 MB)
   - Tour assignments for June 2025
   - Links orders to specific tours and vehicles

3. **`20251015 Sparten.xlsx`** (28 KB)
   - Customer division/category mapping
   - Maps customer billing numbers to product divisions

**File Access**:
- Delivered via SwissTransfer (password protected)
- Available for 7 days from October 15, 2025
- Stored in: `K:\Administration\20_Projekte_und_Pflichtenhefte\Fakturierung\Saison Zahlen 2025` (internal Traveco path)

## Forecasting Architecture

### Core Class: `TravecomForecaster`

The main forecasting system is implemented as a single stateful class in `Forecast.ipynb`:

```python
forecaster = TravecomForecaster(
    data=df,
    target_columns=['total_orders', 'total_km_billed', 'total_km_actual', 'total_tours', 'total_drivers', 'revenue_total', 'external_drivers']
)

# Standard workflow:
monthly_data = forecaster.prepare_data()          # 1. Prepare with temporal features
forecasts = forecaster.generate_forecasts()       # 2. Train all models
fig = forecaster.visualize_forecasts()            # 3. Interactive visualization
fig.show()
```

### Dual Model Strategy

The system implements **two complementary model families**:

#### Model A: Seasonality-Focused
Captures transport industry seasonal patterns (summer peaks, quarterly cycles):

1. **Prophet** (primary)
   - Multiplicative seasonality mode
   - Custom seasonalities: quarterly (91.25 days), monthly (30.5 days)
   - Swiss holiday awareness
   - Config: `yearly_seasonality=True, seasonality_mode='multiplicative', changepoint_prior_scale=0.05`

2. **SARIMAX** (statistical baseline)
   - ARIMA(2,1,2) with seasonal (1,1,1,12) for monthly patterns
   - Branch-specific effects via exogenous variables

3. **XGBoost** (non-linear patterns)
   - 200 estimators, depth=6, learning_rate=0.05
   - Lag features: [1, 3, 6, 12] months
   - Temporal features: month, quarter, week, day_of_year, weekday

#### Model B: Seasonality + Time Decay
Emphasizes recent data relevance (addresses structural market changes):

1. **Weighted Prophet**
   - Exponential decay: `w(t) = exp(-λ * (T - t))` where λ = 0.3-0.5
   - Recent data weighted ~2x older data
   - More responsive changepoint detection (prior_scale=0.1)

2. **Time-Weighted Ensemble**
   - Year-level RandomForest models with exponential year weights
   - Weighted voting: `year_weight = exp(-0.5 * (2025 - year))`

### Data Processing Pipeline

**Key Methods**:

```python
# 1. Temporal Feature Engineering
forecaster.prepare_data()
# Extracts: year, month, week, quarter, day_of_year, weekday
# Aggregates: Monthly by branch
# Calculates: total_orders, total_km, total_drivers, revenue_total, external_drivers sums

# 2. Time Decay Weighting
weights = forecaster.calculate_time_weights(dates, decay_rate=0.3)
# Formula: exp(-decay_rate * days_from_recent / 365)

# 3. Model Training (automated)
forecaster.generate_forecasts()
# Trains all 6 models sequentially with progress bar

# 4. Visualization
forecaster.visualize_forecasts()
# Returns Plotly figure with 3 subplots (one per target metric)
```

## Model Configuration Reference

### Prophet Parameters
```python
Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,      # Monthly aggregation makes this irrelevant
    daily_seasonality=False,       # Same reason
    seasonality_mode='multiplicative',  # Percentage-based seasonality
    changepoint_prior_scale=0.05   # 0.1 for weighted variant (more responsive)
)

# Custom seasonalities
model.add_seasonality(name='quarterly', period=91.25, fourier_order=5)
model.add_seasonality(name='monthly', period=30.5, fourier_order=10)
```

### SARIMAX Parameters
```python
SARIMAX(
    ts_data,
    order=(2, 1, 2),              # ARIMA order: AR(2), I(1), MA(2)
    seasonal_order=(1, 1, 1, 12), # Seasonal: SAR(1), SI(1), SMA(1), period=12
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

### Time Decay Parameters
```python
decay_rate = 0.3        # λ in exponential decay formula
year_decay = 0.5        # For year-level ensemble weighting
```

## Validation and Metrics

### Validation Strategy
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
# Recommended: Hold out 2024 data for validation
```

### Performance Metrics
```python
forecaster.calculate_accuracy_metrics(actual_2025_data)
```

Returns metrics for each target:
- **MAPE** (Mean Absolute Percentage Error) - Primary metric
- **RMSE** (Root Mean Square Error) - Outlier sensitivity
- **Seasonal MAPE** - Month/quarter-specific accuracy
- **Directional Accuracy** - Trend prediction correctness

## Domain Context

### Swiss Transport Industry Specifics
- **LSVA tax**: Heavy vehicle tax affecting costs
- **School vacations**: Demand seasonality driver
- **Cooled goods**: Summer peak seasonality
- **Regional economics**: Branch-specific demand patterns

### Data Scale
- **~1.2M records/year** mentioned in strategic recommendations
- **Monthly aggregation** reduces dimensionality for forecasting
- **Branch-level granularity** maintained for spatial patterns

## Strategic Recommendations

The `information/recommendation.md` file provides implementation guidance:

- **Primary Model**: Prophet for interpretability (board-level presentation)
- **Validation**: Time series CV with 2024 holdout
- **Feature Engineering**: Swiss holidays, weather, fuel prices, LSVA changes
- **Timeline**: 8-week implementation (prep → modeling → validation → presentation)

## Working with This Codebase

### To Modify Models

1. **Adjust seasonality**: Edit Prophet `add_seasonality()` calls in `model_a_prophet()` and `model_b_weighted_prophet()`
2. **Change decay rates**: Modify `decay_rate` parameter in `calculate_time_weights()` and `model_b_weighted_prophet()`
3. **Add lag features**: Update lag list `[1, 3, 6, 12]` in `model_a_xgboost()`
4. **Change target metrics**: Pass different `target_columns` to `TravecomForecaster.__init__()`

### To Add New Models

Add methods following naming pattern:
```python
def model_a_your_model(self, target_col):
    """Model A: Your model description"""
    # Train and return model + forecast
    return model, forecast

def model_b_your_model(self, target_col):
    """Model B: Your model with time decay"""
    # Train with time weighting
    return model, forecast
```

Then integrate into `generate_forecasts()` method.

### To Extend Visualization

Modify `visualize_forecasts()` to add new traces:
```python
fig.add_trace(
    go.Scatter(
        x=your_forecast_dates,
        y=your_forecast_values,
        name='Your Model Name',
        line=dict(color='purple', dash='dash')
    ),
    row=idx, col=1
)
```

## Current Project Status

- **Stage**: Analytical prototype with complete forecasting framework
- **Repository**: `git@github.com:kevinkuhn/customer_traveco.git`
- **Branch**: main (no commits yet - newly created)
- **Missing**: Production deployment code, Docker setup, tests, requirements.txt

**Next Steps for Productionization**:
1. Create `requirements.txt` with pinned versions
2. Extract `TravecomForecaster` into standalone Python module
3. Add data validation and error handling
4. Implement unit tests for model methods
5. Create CLI or API wrapper (FastAPI) for forecast generation
6. Add logging and monitoring
7. Document data preprocessing expectations
