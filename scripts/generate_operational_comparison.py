#!/usr/bin/env python3
"""
Generate comparison forecasts for operational metrics.

- total_orders: Best=XGBoost, generate Seasonal Naive comparison
- revenue_total: Best=Seasonal Naive, generate XGBoost comparison

Appends results to comparison_forecasts.csv
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb

# Paths
BASE_PATH = Path(__file__).parent.parent
DATA_PATH = BASE_PATH / "data" / "processed"
WORKING_DAYS_PATH = BASE_PATH / "data" / "raw" / "TRAVECO_Arbeitstage_2022-laufend_für gopf.com_hb v2.xlsx"

# Metrics that use working days as a feature (based on correlation analysis)
METRICS_WITH_WORKING_DAYS = ['total_orders', 'revenue_total']


def load_operational_data():
    """Load operational time series data."""
    df = pd.read_csv(DATA_PATH / "monthly_aggregated_full_company.csv")
    df['date'] = pd.to_datetime(df['date'])
    return df


def load_working_days() -> pd.DataFrame:
    """Load working days data from Excel file."""
    # Try different header positions to handle different file formats
    for header_row in [0, 2]:
        df_wide = pd.read_excel(WORKING_DAYS_PATH, header=header_row)
        if 'Jahr' in df_wide.columns:
            break
    else:
        raise ValueError(f"Could not find 'Jahr' column in {WORKING_DAYS_PATH}")

    # Map German month names to numbers
    month_map = {
        'Januar': 1, 'Februar': 2, 'März': 3, 'April': 4, 'Mai': 5, 'Juni': 6,
        'Juli': 7, 'August': 8, 'September': 9, 'Oktober': 10, 'November': 11, 'Dezember': 12
    }

    # Transform to long format
    rows = []
    for _, row in df_wide.iterrows():
        year = row['Jahr']
        for month_name, month_num in month_map.items():
            days = row[month_name]
            if pd.notna(days):
                rows.append({'year': int(year), 'month': month_num, 'working_days': int(days)})

    return pd.DataFrame(rows)


def seasonal_naive_forecast(ts, metric, n_periods=13):
    """Generate Seasonal Naive forecast (same month prior year)."""
    last_date = ts['date'].max()
    forecasts = []

    for i in range(1, n_periods + 1):
        forecast_date = last_date + pd.DateOffset(months=i)
        # Get same month from prior year
        prior_year = forecast_date.year - 1
        prior_value = ts[
            (ts['date'].dt.year == prior_year) &
            (ts['date'].dt.month == forecast_date.month)
        ][metric].values

        if len(prior_value) > 0:
            prediction = prior_value[0]
        else:
            prediction = ts[metric].mean()

        forecasts.append({
            'date': forecast_date,
            'prediction': prediction
        })

    return pd.DataFrame(forecasts)


def train_xgboost_forecast(ts, metric, working_days_df, n_periods=13):
    """Train XGBoost and generate forecast with working days feature."""
    df = ts.copy()
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year

    # Merge working days
    use_working_days = metric in METRICS_WITH_WORKING_DAYS
    if use_working_days:
        df = df.merge(working_days_df, on=['year', 'month'], how='left')

    # Create lag features
    for lag in [1, 2, 3, 12]:
        df[f'lag_{lag}'] = df[metric].shift(lag)

    # Rolling features
    df['rolling_3'] = df[metric].rolling(3).mean()
    df['rolling_12'] = df[metric].rolling(12).mean()

    # Drop NaN rows
    df = df.dropna()

    # Features and target
    feature_cols = ['month', 'year', 'lag_1', 'lag_2', 'lag_3', 'lag_12', 'rolling_3', 'rolling_12']
    if use_working_days:
        feature_cols.append('working_days')

    X = df[feature_cols]
    y = df[metric]

    # Train model
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    model.fit(X, y)

    # Generate forecasts
    last_date = ts['date'].max()
    forecasts = []

    # Need to build features iteratively
    history = ts[[metric]].values.flatten().tolist()

    for i in range(1, n_periods + 1):
        forecast_date = last_date + pd.DateOffset(months=i)

        # Build features
        features = {
            'month': forecast_date.month,
            'year': forecast_date.year,
            'lag_1': history[-1],
            'lag_2': history[-2],
            'lag_3': history[-3],
            'lag_12': history[-12] if len(history) >= 12 else np.mean(history),
            'rolling_3': np.mean(history[-3:]),
            'rolling_12': np.mean(history[-12:]) if len(history) >= 12 else np.mean(history)
        }

        # Add working days for future date
        if use_working_days:
            wd = working_days_df[
                (working_days_df['year'] == forecast_date.year) &
                (working_days_df['month'] == forecast_date.month)
            ]['working_days'].values
            features['working_days'] = wd[0] if len(wd) > 0 else working_days_df['working_days'].mean()

        X_pred = pd.DataFrame([features])
        prediction = model.predict(X_pred)[0]

        forecasts.append({
            'date': forecast_date,
            'prediction': prediction
        })

        # Add prediction to history for next iteration
        history.append(prediction)

    return pd.DataFrame(forecasts)


def main():
    print("=" * 70)
    print("Generating Operational Comparison Forecasts (with Working Days)")
    print("=" * 70)

    # Load data
    df = load_operational_data()
    ts = df[df['date'] <= '2025-11-30'].copy()

    # Load working days
    print("\nLoading working days data...")
    working_days_df = load_working_days()
    print(f"  Working days records: {len(working_days_df)} (Jan 2022 - Dec 2026)")

    print(f"\nData range: {ts['date'].min()} to {ts['date'].max()}")
    print(f"Records: {len(ts)}")

    all_forecasts = []

    # 1. Seasonal Naive for total_orders (comparison to XGBoost best)
    # Note: Seasonal Naive doesn't use features like working days
    print("\n1. Generating Seasonal Naive forecast for total_orders...")
    sn_orders = seasonal_naive_forecast(ts, 'total_orders', n_periods=13)
    for _, row in sn_orders.iterrows():
        all_forecasts.append({
            'date': row['date'].strftime('%Y-%m-%d'),
            'metric': 'total_orders',
            'model': 'Seasonal Naive',
            'prediction': row['prediction'],
            'mape': None,  # No MAPE available
            'is_best': False
        })
    print(f"   Generated {len(sn_orders)} forecasts")

    # 2. XGBoost for revenue_total (comparison to Seasonal Naive best)
    # Uses working days as feature (correlation: 0.634)
    print("\n2. Generating XGBoost forecast for revenue_total (with working days)...")
    xgb_revenue = train_xgboost_forecast(ts, 'revenue_total', working_days_df, n_periods=13)
    for _, row in xgb_revenue.iterrows():
        all_forecasts.append({
            'date': row['date'].strftime('%Y-%m-%d'),
            'metric': 'revenue_total',
            'model': 'XGBoost',
            'prediction': row['prediction'],
            'mape': 5.25,  # From model comparison
            'is_best': False
        })
    print(f"   Generated {len(xgb_revenue)} forecasts")

    # Load existing comparison forecasts
    comparison_file = DATA_PATH / "comparison_forecasts.csv"
    existing_df = pd.read_csv(comparison_file)
    print(f"\nExisting comparison forecasts: {len(existing_df)} records")

    # Remove any existing operational forecasts (to avoid duplicates)
    existing_df = existing_df[~existing_df['metric'].isin(['total_orders', 'revenue_total'])]
    print(f"After removing operational: {len(existing_df)} records")

    # Add new forecasts
    new_df = pd.DataFrame(all_forecasts)
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)

    # Save
    combined_df.to_csv(comparison_file, index=False)
    print(f"\nSaved {len(combined_df)} total forecasts to {comparison_file}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary of comparison_forecasts.csv:")
    print("=" * 70)
    for metric in combined_df['metric'].unique():
        models = combined_df[combined_df['metric'] == metric]['model'].unique()
        print(f"  {metric}: {', '.join(models)}")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
