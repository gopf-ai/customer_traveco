# Forecast Validation - Notebook 18

## Overview

**Notebook 18: Forecast Validation** is the most crucial analysis of this entire project. It validates our machine learning forecasting models against actual 2025 data (January-September) and compares them to the traditional budgeting method.

## Purpose

Answers the key business question: **"Are our ML forecasts more accurate than the simple 2024÷12 method?"**

## Three-Way Comparison

The notebook compares three approaches:

1. **Actual (Ground Truth)**: Real 2025 data from January-September
2. **Human/Traditional Method**: 2024 annual total ÷ 12 months (current budgeting practice)
3. **Machine/ML Method**: Predictions from our trained forecasting models

## Data Sources

### 2025 Actual Data (Validation Set)
**Location**: `data/raw/2025/`

**Files**:
- `2025 01 Jan QS Auftragsanalyse.xlsx` through `2025 09 Sep QS Auftragsanalyse.xlsx` (9 monthly order files)
- `2025 QS Tourenaufstellung bis Sept.xlsx` (tour data Jan-Sep)
- `2025 Sparten.xlsx` (customer division mapping)

**CRITICAL**: This data is used ONLY for validation - models are NOT retrained!

### ML Forecasts
**Location**: `data/processed/consolidated_forecast_2025.csv`

Contains best model predictions per metric (from Notebook 14).

### Historical Data (for Human baseline)
**Location**: `data/processed/monthly_aggregated_full_company.csv`

Used to calculate 2024 annual totals for the traditional 2024÷12 method.

## Processing Pipeline

The notebook processes 2025 data through the exact same pipeline as training data:

1. **Data Cleaning** (Notebook 02 logic):
   - Date conversions
   - Filtering rules (exclude Lager orders)
   - Remove invalid records

2. **Feature Engineering** (Notebook 03 logic):
   - Temporal features (year, month, week, quarter)
   - Carrier type identification (internal vs external)
   - Sparten mapping (customer divisions)
   - Betriebszentralen mapping (dispatch centers)

3. **Aggregation** (Notebook 04 logic):
   - Monthly aggregation by Auftraggeber (order owner)
   - Calculate metrics: total_orders, total_km, total_drivers, revenue_total, external_drivers

## Key Metrics Compared

**Primary Business Metrics**:
1. **Total Orders**: Monthly order count
2. **Revenue Total**: Monthly revenue in CHF

## Error Metrics Calculated

For each forecasting method, the notebook calculates:

- **MAPE (Mean Absolute Percentage Error)**: Primary accuracy metric
- **MAE (Mean Absolute Error)**: Absolute deviation in original units
- **Cumulative Error**: Total error over Jan-Sep period
- **Monthly Error %**: Month-by-month percentage deviation

## Plausibility Check

**Section 3** of the notebook validates data quality by comparing:
- 2025 actual data vs 2024 same months
- Ensures changes are within acceptable ranges:
  - ✓ Green: -10% to +10% (normal variance)
  - ⚠️  Yellow: -20% to -10% or +10% to +20% (significant but plausible)
  - ❌ Red: <-20% or >+20% (investigate data quality issues)

This ensures the 2025 data has been processed correctly and is comparable to training data.

## Visualizations Generated

The notebook creates **5 interactive visualizations** saved to `results/`:

### 1. Total Orders Comparison (`forecast_validation_orders_comparison.html`)
- Line chart showing Actual vs Human vs Machine forecasts
- Monthly view from January-September 2025

### 2. Revenue Total Comparison (`forecast_validation_revenue_comparison.html`)
- Line chart showing Actual vs Human vs Machine forecasts for revenue
- Monthly view from January-September 2025

### 3. **THE CRUCIAL ONE**: Error Comparison (`forecast_validation_error_comparison.html`)
- 4-panel dashboard showing side-by-side error comparison
- Top row: Total Orders (Human Error % | Machine Error %)
- Bottom row: Revenue Total (Human Error % | Machine Error %)
- **This is the most important visualization of the entire project**

### 4. Cumulative Error (`forecast_validation_cumulative_error.html`)
- Shows how errors accumulate over time
- Helps identify systematic over/under-prediction

### 5. Error Distribution (`forecast_validation_error_distribution.html`)
- Box plots showing error spread
- Compares variability of Human vs Machine forecasts

## Output Files

**Location**: `results/`

1. `forecast_validation_orders_comparison.html` - Interactive chart
2. `forecast_validation_revenue_comparison.html` - Interactive chart
3. `forecast_validation_error_comparison.html` - **MOST CRUCIAL VISUALIZATION**
4. `forecast_validation_cumulative_error.html` - Interactive chart
5. `forecast_validation_error_distribution.html` - Interactive chart
6. `forecast_validation_summary.csv` - Executive summary table
7. `forecast_validation_monthly_detail.csv` - Detailed monthly data

## Executive Summary

The notebook generates a summary table showing:

| Metric | Method | MAPE (%) | MAE | Cumulative Error |
|--------|--------|----------|-----|------------------|
| Total Orders | Human (2024÷12) | ? | ? | ? |
| Total Orders | Machine (ML) | ? | ? | ? |
| Revenue Total | Human (2024÷12) | ? | ? | ? |
| Revenue Total | Machine (ML) | ? | ? | ? |

**Interpretation**:
- Lower MAPE = Better accuracy
- Lower MAE = Closer predictions on average
- Cumulative Error closer to 0 = Less systematic bias

## Business Impact

This validation demonstrates:

1. **Forecast Accuracy**: How much ML improves over traditional methods
2. **Financial Impact**: Cumulative error shows potential budget variance
3. **Confidence**: Whether ML models are production-ready
4. **ROI**: Justifies investment in forecasting infrastructure

## Usage

**To run the notebook**:

```bash
cd /Users/kk/dev/customer_traveco/notebooks
jupyter notebook 18_forecast_validation_2025.ipynb
```

**Or execute via command line**:

```bash
cd /Users/kk/dev/customer_traveco/notebooks
pipenv run python -c "
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

with open('18_forecast_validation_2025.ipynb', 'r') as f:
    nb = nbformat.read(f, as_version=4)

ep = ExecutePreprocessor(timeout=1800, kernel_name='python3')
ep.preprocess(nb, {'metadata': {'path': '.'}})

with open('18_forecast_validation_2025.ipynb', 'w') as f:
    nbformat.write(nb, f)
"
```

**Expected runtime**: 5-10 minutes (depends on data size)

## Key Insights Generated

The notebook automatically generates insights including:

1. **Accuracy Comparison**: Which method performs better?
2. **Improvement Percentage**: How much better is ML than Human method?
3. **Cumulative Impact**: Total error over 9 months
4. **Recommendations**:
   - Should we adopt ML forecasting?
   - Replace traditional 2024÷12 method?
   - Continue monitoring or refine models?

## Important Notes

- **NO MODEL RETRAINING**: 2025 data is strictly for validation
- **Fair Comparison**: Both methods use same historical data (2022-2024)
- **Apples-to-Apples**: 2025 data processed identically to training data
- **Realistic Baseline**: Human method represents actual current practice

## Next Steps (After Validation)

If ML models outperform:
1. Deploy ML forecasting for Q4 2025 and 2026 planning
2. Replace traditional 2024÷12 method
3. Set up monthly monitoring (actuals vs forecasts)
4. Retrain models quarterly with new data

If ML models underperform:
1. Investigate root causes (model selection, features, data quality)
2. Refine models based on 2025 error patterns
3. Continue using traditional method until improvement
4. Consider hybrid approach (ML for some metrics, traditional for others)

## Date Created

2025-11-03

## Author

Claude Code (automated forecasting system)

## Related Documentation

- `NOTEBOOK_12_FIX.md` - XGBoost model debugging
- `BETRIEBSZENTRALEN_MIGRATION.md` - Dispatch center mapping
- `TROUBLESHOOTING.md` - Common issues
- `information/recommendation.md` - Strategic guidance
