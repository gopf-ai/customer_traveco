#!/usr/bin/env python3
"""
Forecast financial metrics using multiple models.

Workflow:
1. Train models on 2022-2024 data
2. Validate against Jan-Sep 2025 actual data
3. Select best model per metric based on MAPE
4. Generate forecasts for Oct-Dec 2025 + full 2026
5. Compare ML vs Prior Year (same month) methods

Metrics:
- total_revenue (with working days feature)
- personnel_costs
- external_driver_costs
- total_betriebsertrag (with working days feature)
- ebt (with working days feature)

December 2025: Added working days feature per CFO insight.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from datetime import datetime

# ML models
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
import xgboost as xgb

# Configuration
DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "financial_metrics_overview.csv"
WORKING_DAYS_PATH = Path(__file__).parent.parent / "data" / "raw" / "TRAVECO_Arbeitstage_2022-laufend_für gopf.com_hb v2.xlsx"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "processed"
METRICS = ['total_revenue', 'personnel_costs', 'external_driver_costs', 'total_betriebsertrag', 'ebt']

# Metrics that benefit from working days feature (based on correlation analysis)
METRICS_WITH_WORKING_DAYS = ['total_revenue', 'total_betriebsertrag', 'ebt']


def load_data() -> pd.DataFrame:
    """Load and prepare financial metrics data."""
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str) + '-01')
    return df


def load_working_days() -> pd.DataFrame:
    """Load working days data from Excel file."""
    # Try different header positions to handle different file formats
    # v1 format: headers in row 0
    # v2 format: title in row 0, empty row 1, headers in row 2
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


def prepare_time_series(df: pd.DataFrame, metric: str, working_days_df: pd.DataFrame = None) -> pd.DataFrame:
    """Prepare time series data for a specific metric, optionally with working days."""
    ts = df[df['metric'] == metric][['date', 'value', 'year', 'month']].copy()
    ts = ts.sort_values('date').reset_index(drop=True)
    ts = ts.rename(columns={'date': 'ds', 'value': 'y'})

    # Add working days if metric benefits from it
    if working_days_df is not None and metric in METRICS_WITH_WORKING_DAYS:
        ts = ts.merge(working_days_df, on=['year', 'month'], how='left')
        # Fill missing working days with average (for forecasting)
        ts['working_days'] = ts['working_days'].fillna(ts['working_days'].mean())

    return ts


def split_data(ts: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train (2022-2024) and validation (Jan-Nov 2025)."""
    train = ts[ts['ds'] < '2025-01-01'].copy()
    val = ts[(ts['ds'] >= '2025-01-01') & (ts['ds'] <= '2025-11-30')].copy()
    return train, val


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Handle zeros and use absolute values for percentage
    mask = y_true != 0
    if not mask.any():
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def train_prophet(train: pd.DataFrame, metric: str) -> Prophet:
    """Train Prophet model, optionally with working days regressor."""
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.05
    )

    # Add working days as regressor for relevant metrics
    use_working_days = metric in METRICS_WITH_WORKING_DAYS and 'working_days' in train.columns
    if use_working_days:
        model.add_regressor('working_days')

    # Prepare training data
    train_cols = ['ds', 'y'] + (['working_days'] if use_working_days else [])
    model.fit(train[train_cols])
    return model


def train_sarimax(train: pd.DataFrame, metric: str) -> SARIMAX:
    """Train SARIMAX model."""
    # Use simple order for these financial metrics
    model = SARIMAX(
        train['y'],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    fitted = model.fit(disp=False)
    return fitted


def train_xgboost(train: pd.DataFrame, metric: str) -> Tuple[xgb.XGBRegressor, pd.DataFrame]:
    """Train XGBoost model with lag features and optionally working days."""
    df = train.copy()
    df['month'] = df['ds'].dt.month
    df['quarter'] = df['ds'].dt.quarter
    df['xgb_year'] = df['ds'].dt.year

    # Lag features
    for lag in [1, 3, 6, 12]:
        df[f'lag_{lag}'] = df['y'].shift(lag)

    # Rolling features
    df['rolling_mean_3'] = df['y'].rolling(3).mean()
    df['rolling_mean_6'] = df['y'].rolling(6).mean()

    # Drop NaN rows (first 12 months)
    df_clean = df.dropna()

    # Base feature columns
    feature_cols = ['month', 'quarter', 'lag_1', 'lag_3', 'lag_6', 'lag_12',
                    'rolling_mean_3', 'rolling_mean_6']

    # Add working days for relevant metrics
    use_working_days = metric in METRICS_WITH_WORKING_DAYS and 'working_days' in df.columns
    if use_working_days:
        feature_cols.append('working_days')

    X = df_clean[feature_cols]
    y = df_clean['y']

    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
        verbosity=0
    )
    model.fit(X, y)

    # Store whether working days was used
    model.use_working_days = use_working_days

    return model, df


def predict_prophet(model: Prophet, periods: int, last_date: pd.Timestamp,
                    working_days_df: pd.DataFrame = None, metric: str = None) -> pd.DataFrame:
    """Generate Prophet predictions, with working days if applicable."""
    dates = pd.date_range(start=last_date + pd.DateOffset(months=1),
                          periods=periods, freq='MS')
    future = pd.DataFrame({'ds': dates})

    # Add working days if the model uses it
    if metric in METRICS_WITH_WORKING_DAYS and working_days_df is not None:
        future['year'] = future['ds'].dt.year
        future['month'] = future['ds'].dt.month
        future = future.merge(working_days_df, on=['year', 'month'], how='left')
        # Fill missing with average (for future forecasts)
        future['working_days'] = future['working_days'].fillna(working_days_df['working_days'].mean())
        future = future[['ds', 'working_days']]

    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].rename(columns={'yhat': 'prediction'})


def predict_sarimax(model, periods: int, last_date: pd.Timestamp) -> pd.DataFrame:
    """Generate SARIMAX predictions."""
    forecast = model.forecast(periods)
    dates = pd.date_range(start=last_date + pd.DateOffset(months=1),
                          periods=periods, freq='MS')
    return pd.DataFrame({'ds': dates, 'prediction': forecast.values})


def predict_xgboost(model: xgb.XGBRegressor, train_df: pd.DataFrame,
                    periods: int, last_date: pd.Timestamp,
                    working_days_df: pd.DataFrame = None, metric: str = None) -> pd.DataFrame:
    """Generate XGBoost predictions recursively, with working days if applicable."""
    df = train_df.copy()
    predictions = []

    # Base feature columns
    feature_cols = ['month', 'quarter', 'lag_1', 'lag_3', 'lag_6', 'lag_12',
                    'rolling_mean_3', 'rolling_mean_6']

    # Check if model uses working days
    use_working_days = getattr(model, 'use_working_days', False)
    if use_working_days:
        feature_cols.append('working_days')

    for i in range(periods):
        next_date = last_date + pd.DateOffset(months=i+1)

        # Create features for next month
        new_row = {
            'ds': next_date,
            'month': next_date.month,
            'quarter': next_date.quarter,
            'xgb_year': next_date.year
        }

        # Lag features from historical + predicted data
        all_y = list(df['y'].values) + [p['prediction'] for p in predictions]

        if len(all_y) >= 1:
            new_row['lag_1'] = all_y[-1]
        if len(all_y) >= 3:
            new_row['lag_3'] = all_y[-3]
        if len(all_y) >= 6:
            new_row['lag_6'] = all_y[-6]
        if len(all_y) >= 12:
            new_row['lag_12'] = all_y[-12]

        # Rolling features
        if len(all_y) >= 3:
            new_row['rolling_mean_3'] = np.mean(all_y[-3:])
        if len(all_y) >= 6:
            new_row['rolling_mean_6'] = np.mean(all_y[-6:])

        # Add working days if applicable
        if use_working_days and working_days_df is not None:
            wd_row = working_days_df[(working_days_df['year'] == next_date.year) &
                                      (working_days_df['month'] == next_date.month)]
            if len(wd_row) > 0:
                new_row['working_days'] = wd_row['working_days'].values[0]
            else:
                new_row['working_days'] = working_days_df['working_days'].mean()

        # Predict
        X_pred = pd.DataFrame([new_row])[feature_cols]
        pred = model.predict(X_pred)[0]

        predictions.append({'ds': next_date, 'prediction': pred})

    return pd.DataFrame(predictions)


def human_baseline(train: pd.DataFrame) -> float:
    """Calculate human baseline: 2024 total ÷ 12."""
    train_2024 = train[train['ds'].dt.year == 2024]['y'].sum()
    return train_2024 / 12


def same_month_prior_year_baseline(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
    """Calculate baseline: same month from prior year."""
    predictions = []
    for _, row in val.iterrows():
        month = row['ds'].month
        year = row['ds'].year - 1  # Previous year
        prior_value = train[(train['ds'].dt.year == year) &
                            (train['ds'].dt.month == month)]['y'].values
        if len(prior_value) > 0:
            predictions.append(prior_value[0])
        else:
            predictions.append(train['y'].mean())  # Fallback
    return np.array(predictions)


def validate_models(train: pd.DataFrame, val: pd.DataFrame, metric: str,
                    working_days_df: pd.DataFrame = None) -> Dict:
    """Train all models and validate against 2025 data."""
    results = {}

    # Get last training date
    last_train_date = train['ds'].max()
    n_val_periods = len(val)

    # Check if using working days
    use_wd = metric in METRICS_WITH_WORKING_DAYS and 'working_days' in train.columns
    wd_note = " (with working days)" if use_wd else ""

    print(f"\n{'='*60}")
    print(f"Metric: {metric}{wd_note}")
    print(f"Training: {train['ds'].min().strftime('%Y-%m')} to {last_train_date.strftime('%Y-%m')} ({len(train)} months)")
    print(f"Validation: {val['ds'].min().strftime('%Y-%m')} to {val['ds'].max().strftime('%Y-%m')} ({n_val_periods} months)")
    print(f"{'='*60}")

    # Prior Year baseline (same month from prior year)
    prior_year_preds = same_month_prior_year_baseline(train, val)
    prior_year_mape = calculate_mape(val['y'].values, prior_year_preds)
    results['Prior Year'] = {
        'mape': prior_year_mape,
        'predictions': prior_year_preds,
        'model': None
    }
    print(f"  Prior Year:       MAPE = {prior_year_mape:.2f}%  (same month from 2024)")

    # Prophet
    try:
        prophet_model = train_prophet(train, metric)
        prophet_preds = predict_prophet(prophet_model, n_val_periods, last_train_date,
                                        working_days_df, metric)
        prophet_mape = calculate_mape(val['y'].values, prophet_preds['prediction'].values)
        results['Prophet'] = {
            'mape': prophet_mape,
            'predictions': prophet_preds['prediction'].values,
            'model': prophet_model
        }
        print(f"  Prophet:          MAPE = {prophet_mape:.2f}%")
    except Exception as e:
        print(f"  Prophet:          FAILED - {str(e)[:50]}")
        results['Prophet'] = {'mape': np.inf, 'predictions': None, 'model': None}

    # SARIMAX
    try:
        sarimax_model = train_sarimax(train, metric)
        sarimax_preds = predict_sarimax(sarimax_model, n_val_periods, last_train_date)
        sarimax_mape = calculate_mape(val['y'].values, sarimax_preds['prediction'].values)
        results['SARIMAX'] = {
            'mape': sarimax_mape,
            'predictions': sarimax_preds['prediction'].values,
            'model': sarimax_model
        }
        print(f"  SARIMAX:          MAPE = {sarimax_mape:.2f}%")
    except Exception as e:
        print(f"  SARIMAX:          FAILED - {str(e)[:50]}")
        results['SARIMAX'] = {'mape': np.inf, 'predictions': None, 'model': None}

    # XGBoost
    try:
        xgb_model, xgb_train_df = train_xgboost(train, metric)
        xgb_preds = predict_xgboost(xgb_model, xgb_train_df, n_val_periods, last_train_date,
                                    working_days_df, metric)
        xgb_mape = calculate_mape(val['y'].values, xgb_preds['prediction'].values)
        results['XGBoost'] = {
            'mape': xgb_mape,
            'predictions': xgb_preds['prediction'].values,
            'model': xgb_model,
            'train_df': xgb_train_df
        }
        print(f"  XGBoost:          MAPE = {xgb_mape:.2f}%")
    except Exception as e:
        print(f"  XGBoost:          FAILED - {str(e)[:50]}")
        results['XGBoost'] = {'mape': np.inf, 'predictions': None, 'model': None}

    # Select best model
    best_model = min(results.items(), key=lambda x: x[1]['mape'])
    results['best'] = best_model[0]
    print(f"\n  ★ Best model: {best_model[0]} (MAPE: {best_model[1]['mape']:.2f}%)")

    return results


def generate_forecasts(metric: str, model_results: Dict, full_ts: pd.DataFrame,
                       working_days_df: pd.DataFrame = None) -> pd.DataFrame:
    """Generate forecasts for Dec 2025 + 2026 using best model."""
    best_model_name = model_results['best']
    best_result = model_results[best_model_name]

    # Use all data up to Nov 2025 for final forecast
    ts_full = full_ts[full_ts['ds'] <= '2025-11-30'].copy()
    last_date = ts_full['ds'].max()

    # Forecast periods: Dec 2025 (1) + Jan-Dec 2026 (12) = 13 months
    n_periods = 13

    # Check if using working days
    use_wd = metric in METRICS_WITH_WORKING_DAYS and 'working_days' in ts_full.columns
    wd_note = " (with working days)" if use_wd else ""

    print(f"\nGenerating {n_periods}-month forecast using {best_model_name}{wd_note}...")

    if best_model_name == 'Prior Year':
        # Prior Year baseline - use same month from prior year
        dates = pd.date_range(start=last_date + pd.DateOffset(months=1),
                              periods=n_periods, freq='MS')
        predictions = []
        for d in dates:
            # Get same month from prior year
            prior_year = d.year - 1
            prior_value = ts_full[(ts_full['ds'].dt.year == prior_year) &
                                  (ts_full['ds'].dt.month == d.month)]['y'].values
            if len(prior_value) > 0:
                predictions.append(prior_value[0])
            else:
                predictions.append(ts_full['y'].mean())  # Fallback
        forecasts = pd.DataFrame({
            'ds': dates,
            'prediction': predictions
        })

    elif best_model_name == 'Prophet':
        # Retrain Prophet on all available data
        prophet_model = train_prophet(ts_full, metric)
        forecasts = predict_prophet(prophet_model, n_periods, last_date,
                                    working_days_df, metric)

    elif best_model_name == 'SARIMAX':
        # Retrain SARIMAX on all available data
        sarimax_model = train_sarimax(ts_full, metric)
        forecasts = predict_sarimax(sarimax_model, n_periods, last_date)

    elif best_model_name == 'XGBoost':
        # Retrain XGBoost on all available data
        xgb_model, xgb_train_df = train_xgboost(ts_full, metric)
        forecasts = predict_xgboost(xgb_model, xgb_train_df, n_periods, last_date,
                                    working_days_df, metric)

    forecasts['metric'] = metric
    forecasts['model'] = best_model_name
    forecasts['year'] = forecasts['ds'].dt.year
    forecasts['month'] = forecasts['ds'].dt.month

    return forecasts


def main():
    """Main forecasting pipeline."""
    print("=" * 70)
    print("Financial Metrics Forecasting (with Working Days Feature)")
    print("=" * 70)

    # Load data
    df = load_data()
    print(f"\nLoaded {len(df)} records from {DATA_PATH.name}")

    # Load working days data
    working_days_df = load_working_days()
    print(f"Loaded {len(working_days_df)} working days records")
    print(f"Metrics using working days: {', '.join(METRICS_WITH_WORKING_DAYS)}")

    all_results = {}
    all_forecasts = []
    validation_summary = []

    for metric in METRICS:
        # Prepare time series (with working days for relevant metrics)
        ts = prepare_time_series(df, metric, working_days_df)

        if len(ts) == 0:
            print(f"\n⚠️  No data for {metric}, skipping...")
            continue

        # Split data
        train, val = split_data(ts)

        if len(val) == 0:
            print(f"\n⚠️  No 2025 validation data for {metric}, skipping...")
            continue

        # Validate models (pass working_days_df)
        results = validate_models(train, val, metric, working_days_df)
        all_results[metric] = results

        # Store validation summary
        for model_name, model_result in results.items():
            if model_name != 'best' and model_result['predictions'] is not None:
                validation_summary.append({
                    'metric': metric,
                    'model': model_name,
                    'mape': model_result['mape'],
                    'is_best': model_name == results['best']
                })

        # Generate forecasts using best model (pass working_days_df)
        forecasts = generate_forecasts(metric, results, ts, working_days_df)
        all_forecasts.append(forecasts)

    # Combine all forecasts
    df_forecasts = pd.concat(all_forecasts, ignore_index=True)

    # Save forecasts
    forecast_file = OUTPUT_PATH / "financial_metrics_forecasts.csv"
    df_forecasts.to_csv(forecast_file, index=False)
    print(f"\n\n{'='*70}")
    print(f"Forecasts saved to: {forecast_file}")

    # Save validation summary
    df_validation = pd.DataFrame(validation_summary)
    validation_file = OUTPUT_PATH / "financial_metrics_model_comparison.csv"
    df_validation.to_csv(validation_file, index=False)
    print(f"Model comparison saved to: {validation_file}")

    # Print final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")

    print("\n📊 Best Model per Metric:")
    for metric, results in all_results.items():
        best = results['best']
        mape = results[best]['mape']
        print(f"  {metric}: {best} (MAPE: {mape:.2f}%)")

    print("\n📈 Forecast Preview (Oct 2025 - Dec 2026):")
    for metric in METRICS:
        print(f"\n  {metric}:")
        metric_forecasts = df_forecasts[df_forecasts['metric'] == metric]
        for _, row in metric_forecasts.iterrows():
            print(f"    {row['year']}-{row['month']:02d}: {row['prediction']:>15,.0f}")

    # Also show prior year baseline comparison info
    print("\n📊 Prior Year Baseline Info:")
    print("  Uses same month from prior year for predictions")
    print("  (e.g., Oct 2025 prediction = Oct 2024 actual)")

    return df_forecasts, df_validation


if __name__ == "__main__":
    forecasts, validation = main()
