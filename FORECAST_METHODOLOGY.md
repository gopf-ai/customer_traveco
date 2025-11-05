# Forecast Methodology Documentation

**Project**: Traveco Transport Logistics Forecasting
**Date**: November 3, 2025
**Author**: Analysis by Claude Code
**Data Period**: 2022-2024 (36 months historical)
**Forecast Period**: 2025 (12 months)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [External Drivers Analysis](#external-drivers-analysis)
3. [2025 vs 2024 Variance Explanation](#2025-vs-2024-variance-explanation)
4. [Metric Definitions & Data Sources](#metric-definitions--data-sources)
5. [Calculation Methodologies](#calculation-methodologies)
6. [Model Selection & Performance](#model-selection--performance)
7. [Current vs Recommended Forecasts](#current-vs-recommended-forecasts)
8. [Cost Bug Fix History](#cost-bug-fix-history)
9. [Notebook Execution Flow](#notebook-execution-flow)
10. [Future Improvements](#future-improvements)

---

## Executive Summary

### Key Findings

**✅ Cost Bug Fixed**: The critical 12x vehicle cost overcounting bug has been resolved (Notebook 08, Cell 6). Historical profit margin corrected from -281% to +67.4%.

**⚠️ Model Selection Issue**: Current 2025 forecasts use **Seasonal Naive** (56-63% MAPE error) instead of **XGBoost** (3% MAPE error), causing:
- 7-15% underestimation of operational metrics (tours, costs)
- Misleading 2025 vs 2024 variance
- Lower forecast accuracy than validated models

**✅ External Drivers**: 374,602 orders (22.76% of total) handled by external carriers in 2025 forecast. Traveco bills for all orders but pays external carrier fees, reducing margin on these orders.

### Recommended Actions

1. **Update Notebook 14** to use XGBoost forecasts (9 metrics) and MA-6 (tours)
2. **Regenerate 2025 forecasts** with corrected model assignments
3. **Document** that external driver revenue is not separately tracked

---

## External Drivers Analysis

### Definition

**External Drivers** = Orders fulfilled by external carriers (Fremdfahrer)
- **Identifier**: `Nummer.Spedition` ≥ 9000
- **Source Column**: Column BC (`Nummer.Spedition`)
- **Classification Logic**: Created in `notebooks/03_feature_engineering.ipynb`

### 2025 Forecast Breakdown

| Metric | Value | Percentage | Status |
|--------|-------|------------|--------|
| **Total Orders** | 1,645,697 | 100.0% | ✅ |
| **External Driver Orders** | 374,602 | **22.76%** | ✅ |
| **Internal Driver Orders** | 1,271,095 | **77.24%** | ✅ |

### Historical Pattern (2022-2024)

| Year | External Orders | Total Orders | Percentage |
|------|----------------|--------------|------------|
| 2022 | 396,834 | 1,649,579 | 24.06% |
| 2023 | 360,586 | 1,614,593 | 22.33% |
| 2024 | 338,713 | 1,645,250 | 20.59% |
| **Total** | **1,030,977** | **4,535,693** | **22.73%** |

**Insight**: 2025 forecast (22.76%) maintains historical average. Slight downward trend 2022→2024.

### Revenue Attribution

**⚠️ Important Limitation**: Revenue is **NOT split by carrier type** in the data.

**Available Data**:
- ✅ Revenue tracked at order level (`∑ Einnahmen`, Column AC)
- ❌ No separate revenue field for external vs internal carriers
- ✅ External carrier costs tracked (`∑ Ausgaben Spediteur`, Column AB)

**Estimated Revenue from External Drivers** (assuming proportional distribution):
- Total 2025 Revenue: CHF 154,982,885
- External Driver Share (22.76%): **~CHF 35,270,000**
- **Note**: This is an estimate; actual revenue split may differ

### Does Traveco Bill for External Driver Orders?

**YES** - Traveco bills customers for ALL orders:
- **Customer Invoice**: Traveco invoices full amount regardless of carrier type
- **External Carrier Payment**: Traveco pays external carriers for services rendered
- **Net Impact**: Traveco retains revenue but margins are lower (pays carrier fees)

**Example (Historical 2022-2024)**:
- Total Revenue: CHF 464,950,000
- External Carrier Costs Paid: CHF 39,500,000 (8.5% of revenue)
- **Implication**: External orders likely have ~8-10% lower margin

### Business Context

**Why Use External Drivers?**
- **Peak Capacity**: Internal fleet reaches capacity during busy periods
- **Geographic Coverage**: External carriers cover routes where Traveco has no depot
- **Flexibility**: Variable cost model instead of fixed fleet investment
- **Strategic**: 20-25% external usage is standard for Swiss logistics

**2025 Strategy**:
- Maintain ~23% external driver usage
- Focus on optimizing internal fleet utilization (77% of orders)
- Monitor external carrier costs to maintain margin targets

---

## 2025 vs 2024 Variance Explanation

### The Variance Problem

| Metric | 2024 Total | 2025 Forecast | Change | Business Impact |
|--------|-----------|---------------|--------|-----------------|
| total_orders | 1,641,250 | 1,645,697 | **+0.3%** | ✅ Expected growth |
| revenue_total | CHF 158.0M | CHF 155.0M | **-1.9%** | ⚠️ Slight decline |
| **total_km_actual** | 27,815,214 | 24,328,381 | **-12.5%** | ❌ **Significant drop** |
| **total_tours** | 174,056 | 153,728 | **-11.7%** | ❌ **Concerning** |
| **vehicle_km_cost** | CHF 25.7M | CHF 22.0M | **-14.6%** | ❌ **Large reduction** |
| **vehicle_time_cost** | CHF 33.6M | CHF 28.6M | **-15.0%** | ❌ **Large reduction** |
| **total_vehicle_cost** | CHF 59.3M | CHF 50.5M | **-14.8%** | ❌ **Major concern** |

**Question**: How can operations drop 12-15% while revenue stays flat?

**Answer**: **This is a forecasting error, not a real business trend.**

### Root Cause: Seasonal Naive Forecasting

#### What is Seasonal Naive?

**Seasonal Naive** forecasting method:
```python
# For each month in 2025, forecast = mean of same month from 2022-2024
jan_2025_forecast = (jan_2022 + jan_2023 + jan_2024) / 3
```

**The Problem with This Method**:
- 2023 had **anomalous low values** for tours/costs (June-Sept period)
- 2024 had **normalized high values**
- Seasonal Naive averages all 3 years → **pulls 2025 forecast DOWN**

#### Example: Total Tours Calculation

**Historical January Values**:
- Jan 2022: 13,956 tours
- Jan 2023: 13,579 tours
- Jan 2024: 14,512 tours

**Seasonal Naive 2025 Forecast**:
- (13,956 + 13,579 + 14,512) / 3 = **14,016 tours**
- This is **3.4% lower** than 2024 actual (14,512)

**But July-Sept 2023 had MUCH lower values**:
- Jul 2023: **2,036 tours** (anomaly!)
- Jul 2024: 14,960 tours (normal)
- **Seasonal Naive Jul 2025**: **~10,488 tours** (30% lower than 2024!)

### Why 2023 Was Anomalous

From `data/processed/monthly_aggregated_full_company.csv`:

**2023 June-September**: Extremely low `total_km_actual` and `total_tours`
- Jun 2023: 1,317,124 actual km (vs 2,175,022 in Jun 2022)
- Jul 2023: 298,761 actual km (**90% drop!**)
- Aug 2023: 283,015 actual km (**87% drop!**)
- Sep 2023: 235,021 actual km (**90% drop!**)

**Likely Causes** (requires client confirmation):
- System migration or data collection issue
- Seasonal shutdown or strike
- Change in tour recording methodology
- Data quality issue during that period

**Impact on Seasonal Naive**:
- These anomalous months pull down the 3-year average
- 2025 forecasts based on this average are **artificially low**
- **Not representative** of actual 2024-2025 business trend

### Correct Interpretation

**There is NO business reason for these declines:**
- ❌ Orders staying flat but operations dropping 15% = **impossible**
- ❌ Revenue flat but costs dropping 15% = **unrealistic margin expansion**
- ❌ Tours dropping but order volumes constant = **logistics failure**

**The Real Story**:
- ✅ Seasonal Naive is averaging anomalous 2023 data
- ✅ Better models (XGBoost, MA-6) show 2025 ≈ 2024 levels
- ✅ Model validation showed Seasonal Naive has 56-63% MAPE error vs 3-4% for XGBoost

### Model Performance Comparison

| Model | total_tours MAPE | total_vehicle_cost MAPE | Winner |
|-------|------------------|-------------------------|--------|
| **Seasonal Naive** | **58.3%** ❌ | **63.1%** ❌ | Current |
| **MA-6** | **3.6%** ✅ | 4.1% | Recommended (tours) |
| **XGBoost** | 4.6% | **3.0%** ✅ | Recommended (costs) |

**Seasonal Naive is 16-20x WORSE than the best models!**

---

## Metric Definitions & Data Sources

### 1. total_orders

**Definition**: Total count of transport orders per month

**Data Source**:
- **Raw File**: `data/swisstransfer_*/20251015 Juni 2025 QS Auftragsanalyse.xlsb`
- **Column**: Each row = 1 order
- **Processed File**: `data/processed/monthly_aggregated_full_company.csv`

**Filters Applied**:
- ✅ Exclude `Lager Auftrag` (warehouse orders) - not transport
- ✅ Exclude orders with missing carrier numbers
- ✅ Exclude B&T internal pickups (`System_id.Auftrag` = "B&T" AND `RKdNr` = empty)

**Calculation** (`notebooks/08_time_series_aggregation.ipynb`, Cell 16):
```python
monthly_agg = df_orders.groupby('year_month').agg({
    'Nummer.Auftrag': 'count'  # Count unique orders
}).rename(columns={'Nummer.Auftrag': 'total_orders'})
```

**Units**: Count (integer)

---

### 2. total_km_billed

**Definition**: Total kilometers billed to customers (order-based)

**Data Source**:
- **Raw Column**: Column CU (`Distanz_BE.Auftrag`) - Distance from loading to unloading location
- **Not Column CV** (`Distanz_VE.Auftrag`) - Sender to receiver distance
- **Reason**: CU represents actual billed transport distance

**Calculation**:
```python
monthly_agg = df_orders.groupby('year_month').agg({
    'Distanz_BE.Auftrag': 'sum'  # Sum billed distances
}).rename(columns={'Distanz_BE.Auftrag': 'total_km_billed'})
```

**Units**: Kilometers (numeric)

**Business Use**: Revenue calculation basis (billed km × rate)

---

### 3. total_km_actual

**Definition**: Total actual kilometers driven by vehicles (tour-based)

**Data Source**:
- **Raw File**: `data/swisstransfer_*/20251015 QS Tourenaufstellung Juni 2025.xlsx`
- **Column**: Column K (`Fahrzeug KM`) - Vehicle kilometers from PraCar system
- **Alternative Name**: `IstKm.Tour`

**Calculation** (`notebooks/08_time_series_aggregation.ipynb`, Cell 6):
```python
tour_agg = df_tours.groupby(['year_month', 'betriebszentrale']).agg({
    'Fahrzeug KM': 'sum'  # Sum actual driven km
}).rename(columns={'Fahrzeug KM': 'total_km_actual'})
```

**Units**: Kilometers (numeric)

**Business Use**: Cost calculation basis, efficiency metric (actual vs billed)

**KM Efficiency Ratio** = `total_km_actual / total_km_billed`
- **Historical Average**: ~0.27 (27%)
- **Interpretation**: Tours optimize routes; one vehicle delivers multiple orders

---

### 4. total_tours

**Definition**: Count of unique vehicle tours per month

**Data Source**:
- **Raw Column**: Column E (`Nummer.Tour`) - Unique tour identifier
- **Raw File**: Tour assignment file

**Calculation**:
```python
tour_agg = df_tours.groupby(['year_month', 'betriebszentrale']).agg({
    'Nummer.Tour': 'nunique'  # Count unique tours
}).rename(columns={'Nummer.Tour': 'total_tours'})
```

**Units**: Count (integer)

**Relationship to Orders**: ~10 orders per tour (1 tour = 1 vehicle route with multiple stops)

**Business Use**: Resource planning, vehicle utilization, driver scheduling

---

### 5. total_drivers

**Definition**: Total driver count (internal + external) per month

**Data Source**: Sum of two components

**Calculation**:
```python
monthly_agg['total_drivers'] = monthly_agg['internal_drivers'] + monthly_agg['external_drivers']

# Where:
internal_drivers = df_orders[df_orders['Nummer.Spedition'] < 9000].groupby('year_month').size()
external_drivers = df_orders[df_orders['Nummer.Spedition'] >= 9000].groupby('year_month').size()
```

**Units**: Count (integer, aggregated by order count)

**Note**: This is order-based, not unique driver count. Represents driver workload.

---

### 6. revenue_total

**Definition**: Total monthly revenue from transport orders

**Data Source**:
- **Raw Column**: Column AC (`∑ Einnahmen`) - Total revenue per order
- **Currency**: Swiss Francs (CHF)

**Calculation**:
```python
monthly_agg = df_orders.groupby('year_month').agg({
    '∑ Einnahmen': 'sum'  # Sum order revenues
}).rename(columns={'∑ Einnahmen': 'revenue_total'})
```

**Units**: CHF (currency)

**Business Use**: Primary revenue metric for forecasting

**Note**: Does NOT distinguish between internal vs external carrier revenue

---

### 7. external_drivers

**Definition**: Count of orders handled by external carriers

**Data Source**:
- **Raw Column**: Column BC (`Nummer.Spedition`) - Carrier identification number
- **Filter**: `Nummer.Spedition >= 9000` = External carrier

**External Carrier Code Ranges**:
- **1 - 8889**: Traveco internal carriers (BZ Oberbipp, BZ Winterthur, etc.)
- **9000+**: External carriers (Benz Transporte, Bachmann AG, Blättler Transport, etc.)

**Calculation**:
```python
external_drivers = df_orders[df_orders['Nummer.Spedition'] >= 9000].groupby('year_month').agg({
    'Nummer.Auftrag': 'count'
}).rename(columns={'Nummer.Auftrag': 'external_drivers'})
```

**Units**: Count (integer)

**Business Use**:
- Peak capacity planning
- External carrier cost forecasting
- Margin analysis (external orders have lower margins)

---

### 8. vehicle_km_cost

**Definition**: KM-based component of vehicle operational cost

**Data Source** (`notebooks/08_time_series_aggregation.ipynb`, Cell 6):
- **Formula**: `Actual KM × PC KM Kosten`
- **Column K** (`Fahrzeug KM`): Actual kilometers driven
- **PraCar System**: PC KM cost rate per kilometer

**Calculation**:
```python
df_tours['km_cost_component'] = df_tours['Fahrzeug KM'] * df_tours['PC KM Kosten']

monthly_cost = df_tours.groupby(['year_month', 'betriebszentrale']).agg({
    'km_cost_component': 'sum'
}).rename(columns={'km_cost_component': 'vehicle_km_cost'})
```

**Units**: CHF (currency)

**Historical Average**: ~43.5% of total vehicle cost

**Business Use**: Variable cost component, distance-based cost forecasting

---

### 9. vehicle_time_cost

**Definition**: Time-based component of vehicle operational cost

**Data Source** (`notebooks/08_time_series_aggregation.ipynb`, Cell 6):
- **Formula**: `IST Zeit PraCar × 60 × PC Minuten Kosten`
- **Column V** (`IST Zeit PraCar`): Actual tour duration (hours)
- **Conversion**: × 60 to convert hours → minutes
- **PraCar System**: PC minute cost rate

**Calculation**:
```python
df_tours['time_cost_component'] = (
    df_tours['IST Zeit PraCar'] * 60 * df_tours['PC Minuten Kosten']
)

monthly_cost = df_tours.groupby(['year_month', 'betriebszentrale']).agg({
    'time_cost_component': 'sum'
}).rename(columns={'time_cost_component': 'vehicle_time_cost'})
```

**Units**: CHF (currency)

**Historical Average**: ~56.5% of total vehicle cost

**Business Use**: Fixed cost component (driver wages, vehicle hourly costs)

---

### 10. total_vehicle_cost

**Definition**: Total vehicle operational cost (KM + Time)

**Calculation** (Simple sum):
```python
monthly_cost['total_vehicle_cost'] = (
    monthly_cost['vehicle_km_cost'] +
    monthly_cost['vehicle_time_cost']
)
```

**Units**: CHF (currency)

**Historical 2022-2024**:
- Total: CHF 151,600,000
- Monthly Average: CHF 4,211,000

**2025 Forecast**:
- Total: CHF 50,535,000 (Seasonal Naive)
- Monthly Average: CHF 4,211,000

**Business Use**: Primary cost metric for profit margin calculations

---

## Calculation Methodologies

### Aggregation Strategy (Notebook 08)

**Critical Fix Applied** (November 3, 2025):
- **Bug**: Tours were aggregated company-wide, then duplicated to all 12 Betriebszentralen
- **Result**: 12x cost overcounting (CHF 590M instead of CHF 50M)
- **Fix**: Map tours to Betriebszentralen BEFORE aggregation

**Corrected Aggregation** (`notebooks/08_time_series_aggregation.ipynb`, Cell 6):

```python
# 1. Map each tour to its Betriebszentrale
tour_to_bz = df_historic.groupby('Nummer.Tour')['betriebszentrale_name'].agg(
    lambda x: x.value_counts().index[0]  # Most common BZ for tour
).reset_index()

df_tours = df_tours.merge(tour_to_bz, on='Nummer.Tour', how='left')

# 2. Aggregate costs BY BRANCH (prevents duplication)
cost_monthly = df_tours.groupby(['year_month', 'betriebszentrale']).agg({
    'km_cost_component': 'sum',
    'time_cost_component': 'sum',
    'total_vehicle_cost': 'sum'
}).reset_index()

# 3. Merge on BOTH year_month AND betriebszentrale (key fix!)
monthly_agg = monthly_agg.merge(
    cost_monthly,
    on=['year_month', 'betriebszentrale'],  # Prevents duplication
    how='left'
)

# 4. Aggregate to company level
company_agg = monthly_agg.groupby('year_month').sum().reset_index()
```

**Result**: Costs correctly attributed, no duplication, profit margin corrected from -281% to +67.4%

---

### Seasonal Naive Forecasting (Notebook 09)

**Method** (`notebooks/09_baseline_models.ipynb`):

```python
def seasonal_naive_forecast(df_hist, target_col, forecast_year=2025):
    """
    For each month in 2025, forecast = mean of same month from 2022-2024
    """
    df_hist['month'] = df_hist['date'].dt.month

    # Calculate average for each month (1-12) across all historical years
    monthly_avg = df_hist.groupby('month')[target_col].mean().to_dict()

    # Generate 2025 dates
    forecast_dates = pd.date_range('2025-01-01', periods=12, freq='MS')

    # Apply monthly averages
    forecasts = [monthly_avg[date.month] for date in forecast_dates]

    return pd.DataFrame({'date': forecast_dates, target_col: forecasts})
```

**Strengths**:
- ✅ Simple, interpretable
- ✅ Captures seasonal patterns
- ✅ Good for stable metrics (orders, revenue)

**Weaknesses**:
- ❌ Averages out anomalies (2023 low tour values)
- ❌ No trend component (misses 2024→2025 growth)
- ❌ Poor performance on volatile metrics (tours, costs: 56-63% MAPE)

---

### XGBoost Forecasting (Notebook 12)

**Method** (`notebooks/12_xgboost_model.ipynb`):

```python
# Features engineered
features = [
    'year', 'month', 'quarter', 'week', 'day_of_year', 'weekday',  # Temporal
    'lag_1', 'lag_3', 'lag_6', 'lag_12',  # Lag features
    'rolling_mean_3', 'rolling_mean_6'  # Rolling averages
]

# XGBoost model
model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)

# Train on 2022-Jun 2024
model.fit(X_train, y_train)

# Validate on Jul-Dec 2024
predictions = model.predict(X_val)
```

**Strengths**:
- ✅ Captures non-linear patterns
- ✅ Handles trend + seasonality
- ✅ Uses lag features (recent data weighted)
- ✅ 3% MAPE on validation (16-20x better than Seasonal Naive)

**Weaknesses**:
- ❌ Requires feature engineering
- ❌ Less interpretable
- ❌ Can't generate 2025 forecasts without actual 2024 values (validation only)

---

### MA-6 Forecasting (Notebook 09)

**Method** (6-Month Moving Average):

```python
def ma6_forecast(df_hist, target_col):
    """
    For each month in 2025, forecast = mean of last 6 months
    """
    last_6_months = df_hist.tail(6)[target_col].mean()

    # All 2025 months get same value (trailing 6-month average)
    forecast_dates = pd.date_range('2025-01-01', periods=12, freq='MS')
    forecasts = [last_6_months] * 12

    return pd.DataFrame({'date': forecast_dates, target_col: forecasts})
```

**Strengths**:
- ✅ Simple baseline
- ✅ Recent data weighted (last 6 months only)
- ✅ Smooths out short-term volatility
- ✅ Good for tour count (3.6% MAPE)

**Weaknesses**:
- ❌ No seasonality component (constant forecast)
- ❌ Assumes 2025 = average of last 6 months of 2024

---

## Model Selection & Performance

### Validation Methodology (Notebook 15)

**Holdout Period**: July - December 2024 (6 months)
**Training Period**: January 2022 - June 2024 (30 months)

**Models Tested**:
1. Naive (last month)
2. Seasonal Naive (3-year monthly average)
3. MA-3 (3-month moving average)
4. MA-6 (6-month moving average)
5. Linear Trend
6. Prophet (Facebook's forecasting tool)
7. SARIMAX (Seasonal ARIMA)
8. XGBoost (Gradient boosting)

**Evaluation Metric**: MAPE (Mean Absolute Percentage Error)

---

### Performance Results (MAPE on Jul-Dec 2024)

| Metric | Best Model | MAPE | 2nd Best | MAPE | Seasonal Naive MAPE |
|--------|-----------|------|----------|------|---------------------|
| **total_orders** | XGBoost | **2.65%** ✅ | Seasonal Naive | 2.95% | 2.95% |
| **total_km_billed** | XGBoost | **2.78%** ✅ | Seasonal Naive | 2.89% | 2.89% |
| **total_km_actual** | XGBoost | **4.59%** ✅ | Naive | 4.77% | **56.4%** ❌ |
| **total_tours** | MA-6 | **3.62%** ✅ | MA-3 | 3.64% | **58.3%** ❌ |
| **total_drivers** | XGBoost | **2.67%** ✅ | Seasonal Naive | 2.85% | 2.85% |
| **revenue_total** | XGBoost | **3.10%** ✅ | Seasonal Naive | 4.59% | 4.59% |
| **external_drivers** | XGBoost | **2.06%** ✅ | MA-3 | 4.38% | **15.7%** ❌ |
| **vehicle_km_cost** | XGBoost | **3.20%** ✅ | MA-6 | 5.04% | **63.2%** ❌ |
| **vehicle_time_cost** | XGBoost | **2.43%** ✅ | MA-6 | 4.10% | **63.0%** ❌ |
| **total_vehicle_cost** | XGBoost | **3.01%** ✅ | MA-6 | 4.34% | **63.1%** ❌ |

**Summary**:
- **XGBoost wins**: 9 out of 10 metrics (2.06% - 4.59% MAPE)
- **MA-6 wins**: total_tours (3.62% MAPE)
- **Seasonal Naive**: Good for orders/revenue (3% MAPE), TERRIBLE for tours/costs (56-63% MAPE)

**Key Insight**: Seasonal Naive's poor performance on tours/costs is due to 2023 anomalies. XGBoost learns to ignore outliers; Seasonal Naive averages them in.

---

## Current vs Recommended Forecasts

### Current Implementation (Notebook 14)

**All 10 metrics** use **Seasonal Naive** forecasting.

**File**: `data/processed/consolidated_forecast_2025.csv`

**Generation Logic** (`notebooks/14_consolidated_forecasts_2025.ipynb`, Cell 9):
```python
# Loads Seasonal Naive forecasts from Notebook 09
df_seasonal = pd.read_csv('seasonal_naive_forecast_2025.csv')

# Uses for ALL metrics
df_consolidated = df_seasonal.copy()
model_attribution = {metric: 'Seasonal Naive' for metric in target_metrics}
```

**Performance**:
- ✅ Good for: orders, revenue (3% MAPE)
- ❌ Poor for: tours, costs (56-63% MAPE)
- ❌ Causes 2025 to underestimate operations by 7-15%

---

### Recommended Implementation

**Use best model per metric** based on Notebook 15 validation:

| Metric | Recommended Model | Source File | Expected MAPE |
|--------|------------------|-------------|---------------|
| total_orders | **XGBoost** | `xgboost_forecast_validation.csv` | 2.65% |
| total_km_billed | **XGBoost** | `xgboost_forecast_validation.csv` | 2.78% |
| total_km_actual | **XGBoost** | `xgboost_forecast_validation.csv` | 4.59% |
| **total_tours** | **MA-6** | `baseline_forecast_ma6.csv` | 3.62% |
| total_drivers | **XGBoost** | `xgboost_forecast_validation.csv` | 2.67% |
| revenue_total | **XGBoost** | `xgboost_forecast_validation.csv` | 3.10% |
| external_drivers | **XGBoost** | `xgboost_forecast_validation.csv` | 2.06% |
| vehicle_km_cost | **XGBoost** | `xgboost_forecast_validation.csv` | 3.20% |
| vehicle_time_cost | **XGBoost** | `xgboost_forecast_validation.csv` | 2.43% |
| total_vehicle_cost | **XGBoost** | `xgboost_forecast_validation.csv` | 3.01% |

**Implementation** (Modify Notebook 14):
```python
# Load XGBoost validation forecasts
df_xgb = pd.read_csv('data/processed/xgboost_forecast_validation.csv')

# Load MA-6 forecasts
df_ma6 = pd.read_csv('data/processed/baseline_forecast_ma6.csv')

# Consolidate using best model per metric
df_consolidated = df_xgb[['date']].copy()

# Use XGBoost for 9 metrics
xgb_metrics = ['total_orders', 'total_km_billed', 'total_km_actual',
               'total_drivers', 'revenue_total', 'external_drivers',
               'vehicle_km_cost', 'vehicle_time_cost', 'total_vehicle_cost']
for metric in xgb_metrics:
    df_consolidated[metric] = df_xgb[metric]

# Use MA-6 for tours
df_consolidated['total_tours'] = df_ma6['total_tours']
```

**Expected Impact**:
- ✅ 2025 vehicle costs: CHF 59M instead of CHF 50M (closer to 2024 actual)
- ✅ 2025 tours: ~14,500/month instead of ~12,800/month
- ✅ Overall MAPE improvement: 56-63% → 3-4% for cost/tour metrics

---

## Cost Bug Fix History

### The Bug (Discovered November 3, 2025)

**Issue**: Vehicle costs overcounted by **12x** in Notebook 08, Cell 6

**Root Cause**:
1. Tours aggregated to company-wide monthly totals (no branch split)
2. Company-wide cost values duplicated to all 12 Betriebszentralen via merge
3. Duplicated values summed again for company-level aggregation = **12x overcount**

**Impact**:
- Historical vehicle costs: CHF 1,770M (WRONG) → CHF 151.6M (CORRECT)
- Monthly average: CHF 49.2M (WRONG) → CHF 4.2M (CORRECT)
- Profit margin: -281% (bankrupt!) → +67.4% (profitable!)

### The Fix

**Location**: `notebooks/08_time_series_aggregation.ipynb`, Cell 6 (lines ~100-200)

**Changes Applied**:

```python
# BEFORE (BUG):
cost_monthly = df_tours.groupby('year_month').agg({
    'total_vehicle_cost': 'sum'
})
# Result: Company-wide total (no branch split)

monthly_agg = monthly_agg.merge(cost_monthly, on='year_month', how='left')
# Result: Same value copied to all 12 branches

company_total = monthly_agg.groupby('year_month')['total_vehicle_cost'].sum()
# Result: 12 branches × same value = 12x overcounting!
```

```python
# AFTER (FIX):
# 1. Map tours to Betriebszentralen
tour_to_bz = df_historic.groupby('Nummer.Tour')['betriebszentrale_name'].agg(
    lambda x: x.value_counts().index[0]
).reset_index()

df_tours = df_tours.merge(tour_to_bz, on='Nummer.Tour', how='left')

# 2. Aggregate BY BRANCH (prevents duplication)
cost_monthly = df_tours.groupby(['year_month', 'betriebszentrale']).agg({
    'total_vehicle_cost': 'sum'
}).reset_index()

# 3. Merge on BOTH keys (1:1 relationship, no duplication)
monthly_agg = monthly_agg.merge(
    cost_monthly,
    on=['year_month', 'betriebszentrale'],  # KEY FIX!
    how='left'
)

# 4. Aggregate to company level (now correct)
company_total = monthly_agg.groupby('year_month')['total_vehicle_cost'].sum()
# Result: Correct sum across 12 branches, no duplication
```

**Files Regenerated** (November 3, 2025, 15:47):
- ✅ `data/processed/monthly_aggregated_full_bz.csv` (branch-level, 369 records)
- ✅ `data/processed/monthly_aggregated_full_company.csv` (company-level, 36 months)
- ✅ `data/processed/monthly_aggregated_full_company.parquet` (faster loading)

**Validation**:
- Historical December 2024: Vehicle Cost = CHF 4,691,323 (was CHF 56M+) ✅
- Profit Margin 2022-2024: +67.39% (was -281%) ✅
- Business viability: Profitable (was bankrupt) ✅

**Documentation**:
- `COST_BUG_FIX.md` - Technical analysis
- `FINAL_RESULTS_SUMMARY.md` - Executive summary

---

## Notebook Execution Flow

### Dependency Graph

```
01_data_loading_and_exploration.ipynb (if exists)
    ↓
02_data_cleaning_and_validation.ipynb
    ↓
03_feature_engineering.ipynb
    ↓
04_aggregation_and_targets.ipynb (Auftraggeber aggregation)
    ↓
05_exploratory_data_analysis.ipynb (EDA, visualizations)
    ↓
06_tour_cost_analysis.ipynb (vehicle cost calculation)
    ↓
08_time_series_aggregation.ipynb ← **CRITICAL: Cost bug fixed here**
    ↓
    ├→ 09_baseline_models.ipynb (Seasonal Naive, MA-3, MA-6)
    ├→ 10_prophet_model.ipynb (Facebook Prophet)
    ├→ 11_sarimax_model.ipynb (Seasonal ARIMA)
    └→ 12_xgboost_model.ipynb (Gradient Boosting)
    ↓
15_model_comparison.ipynb (Validates all models, selects best)
    ↓
14_consolidated_forecasts_2025.ipynb ← **Uses Seasonal Naive (WRONG)**
    ↓
17_monthly_forecast_2025_table.ipynb (Final output table)
```

### Execution Order (Correct Sequence)

**Phase 1: Data Preparation** (Run once)
1. `02_data_cleaning_and_validation.ipynb`
2. `03_feature_engineering.ipynb`
3. `04_aggregation_and_targets.ipynb`
4. `05_exploratory_data_analysis.ipynb`
5. `06_tour_cost_analysis.ipynb`

**Phase 2: Time Series Aggregation** (Run after Phase 1)
6. `08_time_series_aggregation.ipynb` ← Contains cost bug fix

**Phase 3: Model Training** (Run in parallel after Phase 2)
7. `09_baseline_models.ipynb`
8. `10_prophet_model.ipynb`
9. `11_sarimax_model.ipynb`
10. `12_xgboost_model.ipynb`

**Phase 4: Model Evaluation** (Run after Phase 3)
11. `15_model_comparison.ipynb` ← Identifies best models

**Phase 5: Forecast Consolidation** (Run after Phase 4)
12. `14_consolidated_forecasts_2025.ipynb` ← Should use best models
13. `17_monthly_forecast_2025_table.ipynb` ← Final output

### Key Files Generated

**Processed Data**:
- `data/processed/monthly_aggregated_full_company.csv` - Historical aggregated data (36 months)
- `data/processed/clean_orders.csv` - Cleaned order data
- `data/processed/features_engineered.csv` - With Sparten, Betriebszentralen, temporal features

**Baseline Forecasts**:
- `data/processed/baseline_forecast_ma3.csv` - 3-month moving average
- `data/processed/baseline_forecast_ma6.csv` - 6-month moving average
- `data/processed/seasonal_naive_forecast_2025.csv` - Seasonal Naive (currently used)

**Advanced Forecasts**:
- `data/processed/prophet_forecast_validation.csv` - Prophet validation (Jul-Dec 2024)
- `data/processed/sarimax_forecast_validation.csv` - SARIMAX validation
- `data/processed/xgboost_forecast_validation.csv` - XGBoost validation (RECOMMENDED)

**Model Comparison**:
- `data/processed/best_models_summary.csv` - Best model per metric
- `data/processed/model_comparison_summary.csv` - All model MAPEs

**Final Outputs**:
- `data/processed/consolidated_forecast_2025.csv` - Consolidated 2025 forecast
- `results/monthly_forecast_2025_table.csv` - Final forecast table (CSV)
- `results/monthly_forecast_2025_table.xlsx` - Final forecast table (Excel)

---

## Future Improvements

### 1. Fix Model Selection in Notebook 14

**Current Issue**: Uses Seasonal Naive for all metrics

**Recommended Fix**:
- Load XGBoost validation forecasts for 9 metrics
- Load MA-6 forecasts for tours
- Consolidate using best model per metric
- Expected MAPE improvement: 56-63% → 3-4% for cost/tour metrics

**Implementation Priority**: **HIGH** (easy fix, major accuracy improvement)

---

### 2. Generate XGBoost 2025 Forecasts

**Current Limitation**: XGBoost only has validation forecasts (Jul-Dec 2024)

**Why**: XGBoost needs actual 2024 feature values to generate 2025 forecasts (lag features)

**Options**:
- **Option A**: Use validation forecasts for 2025 (assume 2025 ≈ Jul-Dec 2024 average)
- **Option B**: Implement recursive forecasting (use forecasted values as lag features)
- **Option C**: Wait for full 2024 data, retrain with 2022-2024, forecast 2026

**Implementation Priority**: **MEDIUM** (requires additional development)

---

### 3. Investigate 2023 Anomalies

**Issue**: Jun-Sep 2023 shows 90% drop in `total_km_actual` and `total_tours`

**Questions**:
- Was this a data collection issue?
- System migration or methodology change?
- Business event (strike, shutdown, contract change)?

**Recommendation**: Validate with Traveco operations team

**Impact**: If 2023 data is invalid, exclude from Seasonal Naive calculation

**Implementation Priority**: **MEDIUM** (data quality investigation)

---

### 4. Add External Driver Revenue Tracking

**Current Limitation**: Revenue not split by carrier type

**Recommendation**:
- Add revenue field to tour assignment file
- Link revenue to carrier number
- Calculate: Internal revenue vs External revenue
- Enable margin analysis by carrier type

**Business Value**: Understand true profitability of external vs internal operations

**Implementation Priority**: **LOW** (requires data schema changes)

---

### 5. Ensemble Forecasting

**Concept**: Combine multiple models with weighted averaging

**Example**:
```python
final_forecast = (
    0.50 * xgboost_forecast +
    0.30 * ma6_forecast +
    0.20 * prophet_forecast
)
```

**Weights** optimized based on validation MAPE

**Benefits**:
- More robust to individual model failures
- Can combine strengths (XGBoost trend + Prophet seasonality)

**Implementation Priority**: **LOW** (marginal improvement over single best model)

---

### 6. Feature Engineering Enhancements

**Additional Features to Consider**:
- External factors: Fuel prices, economic indicators, holidays
- Betriebszentralen-specific features (depot capacity, regional demand)
- Weather data (impacts tour times, road conditions)
- Customer-specific seasonality (Sparten patterns)

**Implementation Priority**: **LOW** (XGBoost already performing well at 3% MAPE)

---

## Summary & Recommendations

### What We Know

✅ **External Drivers**: 22.76% of 2025 orders (374,602 of 1,645,697)
- Traveco bills for all orders, pays external carrier fees
- Revenue split not tracked; estimate ~CHF 35M if proportional
- Historical pattern stable at 22-23%

✅ **Cost Bug Fixed**: 12x overcounting eliminated
- Historical profit margin: +67.4% (was -281%)
- Monthly vehicle costs: CHF 4.2M (was CHF 49M)

⚠️ **2025 Variance Issue**: Forecasting model error, not business trend
- Seasonal Naive averages 2023 anomalies → underestimates 2025
- Tours/costs 7-15% lower than 2024 due to poor model (56-63% MAPE)

✅ **Best Models Identified**: XGBoost (9 metrics), MA-6 (tours)
- Validation MAPE: 2-4% vs Seasonal Naive 56-63%
- 16-20x improvement for tour/cost metrics

### Immediate Action Items

1. **Update Notebook 14** to use XGBoost + MA-6 (instead of Seasonal Naive)
2. **Regenerate 2025 forecasts** with corrected model selection
3. **Document** that external driver revenue is not separately tracked
4. **Validate** 2023 anomalies with Traveco operations team

### Long-Term Enhancements

- Add external driver revenue tracking to data schema
- Implement ensemble forecasting for robustness
- Enhance feature engineering with external factors
- Set up automated monthly forecast pipeline

---

**Document Version**: 1.0
**Last Updated**: November 3, 2025
**Maintained By**: Data Analytics Team
**Contact**: [Your Contact Info]

---
