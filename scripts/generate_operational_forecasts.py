#!/usr/bin/env python3
"""
Generate operational forecasts using best-performing models per metric.

Best Models (validated on 2025 actuals):
- total_orders: XGBoost (MAPE: 3.60%)
- revenue_total: Seasonal Naive (MAPE: 4.10%)

Output: Dec 2025 - Dec 2026 forecasts
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# Paths
BASE_PATH = Path(__file__).parent.parent
DATA_PATH = BASE_PATH / "data" / "processed"

def load_historical_data():
    """Load historical operational data (2022-Nov 2025)."""
    df = pd.read_csv(DATA_PATH / "monthly_aggregated_full_company.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df

def create_features(df, target_col):
    """Create features for XGBoost model."""
    df = df.copy()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter

    # Lag features (same month last year, previous month)
    df['lag_12'] = df[target_col].shift(12)  # Same month last year
    df['lag_1'] = df[target_col].shift(1)    # Previous month
    df['lag_2'] = df[target_col].shift(2)
    df['lag_3'] = df[target_col].shift(3)

    # Rolling features
    df['rolling_3m_mean'] = df[target_col].shift(1).rolling(3).mean()
    df['rolling_12m_mean'] = df[target_col].shift(1).rolling(12).mean()

    # YoY growth
    df['yoy_growth'] = (df[target_col] - df['lag_12']) / df['lag_12']

    return df

def train_xgboost(df, target_col):
    """Train XGBoost model for orders."""
    df_features = create_features(df, target_col)

    # Drop rows with NaN (first 12 months due to lag features)
    df_train = df_features.dropna()

    feature_cols = ['year', 'month', 'quarter', 'lag_12', 'lag_1', 'lag_2', 'lag_3',
                   'rolling_3m_mean', 'rolling_12m_mean']

    X = df_train[feature_cols]
    y = df_train[target_col]

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    model.fit(X, y)

    return model, feature_cols, df_features

def predict_xgboost(model, feature_cols, df_features, target_col, forecast_dates):
    """Generate XGBoost predictions for forecast period."""
    predictions = []
    df_pred = df_features.copy()

    for date in forecast_dates:
        # Create row for this date
        new_row = pd.DataFrame({
            'date': [date],
            target_col: [np.nan]
        })
        new_row['year'] = date.year
        new_row['month'] = date.month
        new_row['quarter'] = (date.month - 1) // 3 + 1

        # Get lag values from existing data
        lag_12_date = date - pd.DateOffset(months=12)
        lag_1_date = date - pd.DateOffset(months=1)
        lag_2_date = date - pd.DateOffset(months=2)
        lag_3_date = date - pd.DateOffset(months=3)

        new_row['lag_12'] = df_pred[df_pred['date'] == lag_12_date][target_col].values[0] if len(df_pred[df_pred['date'] == lag_12_date]) > 0 else np.nan
        new_row['lag_1'] = df_pred[df_pred['date'] == lag_1_date][target_col].values[0] if len(df_pred[df_pred['date'] == lag_1_date]) > 0 else np.nan
        new_row['lag_2'] = df_pred[df_pred['date'] == lag_2_date][target_col].values[0] if len(df_pred[df_pred['date'] == lag_2_date]) > 0 else np.nan
        new_row['lag_3'] = df_pred[df_pred['date'] == lag_3_date][target_col].values[0] if len(df_pred[df_pred['date'] == lag_3_date]) > 0 else np.nan

        # Rolling means
        recent_3 = df_pred[df_pred['date'] < date].tail(3)[target_col].mean()
        recent_12 = df_pred[df_pred['date'] < date].tail(12)[target_col].mean()
        new_row['rolling_3m_mean'] = recent_3
        new_row['rolling_12m_mean'] = recent_12

        # Predict
        X_pred = new_row[feature_cols]
        pred = model.predict(X_pred)[0]
        predictions.append(pred)

        # Add prediction to df for next iteration's lag
        new_row[target_col] = pred
        df_pred = pd.concat([df_pred, new_row], ignore_index=True)

    return predictions

def calculate_seasonal_naive_with_trend(df, target_col, forecast_dates):
    """
    Calculate Seasonal Naive forecasts with trend adjustment.

    Seasonal Naive: Use same month from prior year as base
    Trend: Apply average YoY growth rate
    """
    df = df.copy()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    # Calculate YoY growth for each month we have data for
    yoy_changes = []
    for year in [2023, 2024, 2025]:
        for month in range(1, 13):
            current = df[(df['year'] == year) & (df['month'] == month)][target_col].values
            prior = df[(df['year'] == year - 1) & (df['month'] == month)][target_col].values
            if len(current) > 0 and len(prior) > 0:
                yoy_change = (current[0] - prior[0]) / prior[0]
                yoy_changes.append(yoy_change)

    # Average YoY growth rate
    avg_yoy_growth = np.mean(yoy_changes) if yoy_changes else 0.02
    print(f"  Average YoY growth rate for {target_col}: {avg_yoy_growth:.2%}")

    predictions = []
    for date in forecast_dates:
        # Get same month from prior year
        prior_year_date = date - pd.DateOffset(years=1)
        prior_value = df[df['date'] == prior_year_date][target_col].values

        if len(prior_value) > 0:
            # Apply trend adjustment
            pred = prior_value[0] * (1 + avg_yoy_growth)
        else:
            # Fallback: use average of same month across years
            month = date.month
            same_month_values = df[df['month'] == month][target_col].values
            pred = np.mean(same_month_values) * (1 + avg_yoy_growth)

        predictions.append(pred)

    return predictions

def main():
    print("=" * 70)
    print("Generating Operational Forecasts (Dec 2025 - Dec 2026)")
    print("=" * 70)

    # Load data
    print("\n1. Loading historical data...")
    df = load_historical_data()
    print(f"   Loaded {len(df)} months of data ({df['date'].min()} to {df['date'].max()})")

    # Define forecast period
    forecast_dates = pd.date_range('2025-12-01', '2026-12-01', freq='MS')
    print(f"\n2. Forecast period: {forecast_dates[0].strftime('%Y-%m')} to {forecast_dates[-1].strftime('%Y-%m')}")

    # Train XGBoost for orders
    print("\n3. Training XGBoost for total_orders...")
    xgb_model_orders, feature_cols_orders, df_features_orders = train_xgboost(df, 'total_orders')
    orders_forecasts = predict_xgboost(xgb_model_orders, feature_cols_orders, df_features_orders, 'total_orders', forecast_dates)
    print(f"   Generated {len(orders_forecasts)} forecasts for orders")

    # Calculate Seasonal Naive for revenue (best model)
    print("\n4. Calculating Seasonal Naive for revenue_total (best model)...")
    revenue_forecasts = calculate_seasonal_naive_with_trend(df, 'revenue_total', forecast_dates)
    print(f"   Generated {len(revenue_forecasts)} forecasts for revenue (Seasonal Naive)")

    # Train XGBoost for revenue (for comparison)
    print("\n5. Training XGBoost for revenue_total (comparison)...")
    xgb_model_revenue, feature_cols_revenue, df_features_revenue = train_xgboost(df, 'revenue_total')
    revenue_xgboost_forecasts = predict_xgboost(xgb_model_revenue, feature_cols_revenue, df_features_revenue, 'revenue_total', forecast_dates)
    print(f"   Generated {len(revenue_xgboost_forecasts)} forecasts for revenue (XGBoost)")

    # Create forecast dataframe
    print("\n6. Creating forecast dataset...")
    forecast_df = pd.DataFrame({
        'date': forecast_dates,
        'total_orders': orders_forecasts,
        'revenue_total': revenue_forecasts,
        'source': 'forecast'
    })

    # Combine with historical 2025 actuals
    df_2025 = df[df['date'].dt.year == 2025].copy()

    # Create combined 2025-2026 dataset
    combined = pd.concat([df_2025, forecast_df], ignore_index=True)
    combined = combined.sort_values('date').reset_index(drop=True)

    # Save main forecast file
    output_file = DATA_PATH / "combined_forecast_2025_2026.csv"
    combined.to_csv(output_file, index=False)
    print(f"\n   Saved: {output_file}")

    # Save comparison forecasts (XGBoost for revenue)
    comparison_df = pd.DataFrame({
        'date': forecast_dates,
        'metric': 'revenue_total',
        'model': 'XGBoost',
        'prediction': revenue_xgboost_forecasts
    })
    comparison_file = DATA_PATH / "comparison_forecasts.csv"
    comparison_df.to_csv(comparison_file, index=False)
    print(f"   Saved: {comparison_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("FORECAST SUMMARY")
    print("=" * 70)

    print("\n📊 TOTAL ORDERS (XGBoost)")
    print("-" * 50)
    for i, (date, orders) in enumerate(zip(forecast_dates, orders_forecasts)):
        if i < 3 or i >= len(forecast_dates) - 2:
            print(f"   {date.strftime('%Y-%m')}: {orders:,.0f}")
        elif i == 3:
            print("   ...")

    print(f"\n   2026 Total: {sum(orders_forecasts):,.0f}")

    print("\n📊 REVENUE (Seasonal Naive)")
    print("-" * 50)
    for i, (date, rev) in enumerate(zip(forecast_dates, revenue_forecasts)):
        if i < 3 or i >= len(forecast_dates) - 2:
            print(f"   {date.strftime('%Y-%m')}: CHF {rev:,.0f}")
        elif i == 3:
            print("   ...")

    print(f"\n   2026 Total: CHF {sum(revenue_forecasts):,.0f}")

    # Compare to 2025 actuals
    print("\n📈 COMPARISON TO 2025")
    print("-" * 50)
    orders_2025 = df_2025['total_orders'].sum()
    orders_2026 = sum(orders_forecasts[1:])  # Exclude Dec 2025
    print(f"   Orders 2025: {orders_2025:,.0f}")
    print(f"   Orders 2026 (forecast): {orders_2026:,.0f}")
    print(f"   Change: {(orders_2026/orders_2025 - 1)*100:+.1f}%")

    revenue_2025 = df_2025['revenue_total'].sum()
    revenue_2026 = sum(revenue_forecasts[1:])  # Exclude Dec 2025
    print(f"\n   Revenue 2025: CHF {revenue_2025:,.0f}")
    print(f"   Revenue 2026 (forecast): CHF {revenue_2026:,.0f}")
    print(f"   Change: {(revenue_2026/revenue_2025 - 1)*100:+.1f}%")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)

    return combined

if __name__ == "__main__":
    combined = main()
