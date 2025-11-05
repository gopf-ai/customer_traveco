# Notebook 12 XGBoost Model - Debug Fix

## Problem
Notebook 12 was failing with `KeyError: 'total_tours'` because it tried to train models on metrics that don't exist in the regenerated data from Notebook 08.

## Root Cause
After fixing Notebook 08's type conversion bug, the regenerated `monthly_aggregated_full_company.csv` only contains order-based metrics:
- `total_orders`
- `total_km` (not `total_km_billed`)
- `total_drivers`
- `external_drivers`
- `revenue_total`

But Notebook 12's `target_metrics` list included 10 metrics, including tour-based metrics that don't exist:
- `total_km_billed` ❌ (has `total_km` instead)
- `total_km_actual` ❌
- `total_tours` ❌
- `vehicle_km_cost` ❌
- `vehicle_time_cost` ❌
- `total_vehicle_cost` ❌

## Solution Applied

### Fix 1: Column Filtering Logic (Cell-5)
Added automatic column filtering to only train on metrics that actually exist:

```python
# Backward compatibility check - handle column name changes
if 'total_km' in df.columns and 'total_km_billed' not in df.columns:
    target_metrics = [m.replace('total_km_billed', 'total_km') if m == 'total_km_billed' else m for m in target_metrics]

# Filter to only include columns that actually exist in the dataframe
available_metrics = [m for m in target_metrics if m in df.columns]
missing_metrics = [m for m in target_metrics if m not in df.columns]

if missing_metrics:
    print(f"⚠️  The following metrics are not available in the dataset and will be skipped:")
    for m in missing_metrics:
        print(f"   - {m}")
    print(f"\n✓ Training models for {len(available_metrics)} available metrics:")
    for m in available_metrics:
        print(f"   - {m}")

target_metrics = available_metrics
```

This ensures:
- Only available columns are used for training
- `total_km_billed` → `total_km` mapping works correctly
- User gets clear warning about skipped metrics

### Fix 2: Removed Problematic Cell-22
Deleted cell-22 which tried to consolidate 2025 forecasts BEFORE they were generated (cell-23). The cell referenced `xgb_forecasts_2025` which didn't exist yet.

## Result
Notebook 12 now:
✅ Trains only on available metrics (total_orders, total_km, total_drivers, external_drivers, revenue_total)
✅ Skips missing metrics with clear warnings
✅ Handles backward compatibility for column name changes
✅ Executes without errors

## Future Work
To enable full 10-metric forecasting:
1. Update Notebook 08 to generate tour-based metrics from tour data
2. Calculate `total_tours`, `total_km_actual` from tour assignments
3. Calculate vehicle costs (`vehicle_km_cost`, `vehicle_time_cost`, `total_vehicle_cost`)
4. Re-run Notebook 08 to regenerate data with all 10 metrics
5. Re-run Notebooks 09-17 to regenerate forecasts

## Files Modified
- `/Users/kk/dev/customer_traveco/notebooks/12_xgboost_model.ipynb`
  - Updated cell-5: Added column filtering logic
  - Removed cell-22: Deleted premature consolidation code

## Date
2025-11-03
