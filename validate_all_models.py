"""
Multi-Model Forecast Validation Script

Validates all forecasting models against actual 2025 data (Jan-Sep).
Generates comprehensive comparison tables and rankings.

Models compared:
- Human (2024÷12)
- Seasonal Naive
- XGBoost
- CatBoost
- LightGBM
- Ensemble (Best Model)
- Ensemble (Weighted)
- Ensemble (Hybrid)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print('='*80)
print('MULTI-MODEL FORECAST VALIDATION')
print('='*80)

# Load actual 2025 data (processed in notebook 18)
print('\n1. Loading actual 2025 data...')
actual_file = Path('data/processed/2025_actual_monthly.csv')

if not actual_file.exists():
    # Need to extract from notebook 18 execution
    print('   Extracting actual data from validation results...')
    validation_detail = pd.read_csv('results/forecast_validation_monthly_detail.csv')
    df_actual = validation_detail[['date', 'orders_actual', 'revenue_actual']].copy()
    df_actual.columns = ['date', 'total_orders', 'revenue_total']
    df_actual['date'] = pd.to_datetime(df_actual['date'])
else:
    df_actual = pd.read_csv(actual_file)
    df_actual['date'] = pd.to_datetime(df_actual['date'])

print(f'   ✓ Loaded: {len(df_actual)} months')

# Load all model predictions
print('\n2. Loading all model predictions...')
df_predictions = pd.read_csv('data/processed/all_model_predictions_2025_validation.csv')
df_predictions['date'] = pd.to_datetime(df_predictions['date'])
print(f'   ✓ Loaded: {len(df_predictions)} months × {len(df_predictions.columns)-1} predictions')

# Define metrics and models
focus_metrics = ['total_orders', 'revenue_total']
models = ['human', 'seasonal_naive', 'xgboost', 'catboost', 'lightgbm',
          'ensemble_best', 'ensemble_weighted', 'ensemble_hybrid']

model_names_clean = {
    'human': 'Human (2024÷12)',
    'seasonal_naive': 'Seasonal Naive',
    'xgboost': 'XGBoost',
    'catboost': 'CatBoost',
    'lightgbm': 'LightGBM',
    'ensemble_best': 'Ensemble (Best)',
    'ensemble_weighted': 'Ensemble (Weighted)',
    'ensemble_hybrid': 'Ensemble (Hybrid 60/40)'
}

# Calculate validation metrics for all models
print('\n3. Calculating validation metrics...')
print('='*80)

results = []

for metric in focus_metrics:
    print(f'\n{metric.upper()}:')
    print('-'*80)

    actual_values = df_actual[metric].values

    for model in models:
        col_name = f'{metric}_{model}'

        if col_name not in df_predictions.columns:
            print(f'   ⚠️  {model_names_clean[model]:30s}: NOT AVAILABLE')
            continue

        predicted_values = df_predictions[col_name].values

        # Calculate metrics
        errors = predicted_values - actual_values
        abs_errors = np.abs(errors)
        pct_errors = (errors / actual_values) * 100
        abs_pct_errors = np.abs(pct_errors)

        mae = abs_errors.mean()
        mape = abs_pct_errors.mean()
        rmse = np.sqrt((errors ** 2).mean())
        cumulative_error = errors.sum()

        results.append({
            'Metric': metric,
            'Model': model_names_clean[model],
            'MAPE (%)': mape,
            'MAE': mae,
            'RMSE': rmse,
            'Cumulative Error': cumulative_error,
            'Avg Prediction': predicted_values.mean(),
            'Actual Avg': actual_values.mean()
        })

        print(f'   {model_names_clean[model]:30s}: MAPE={mape:6.2f}%, MAE={mae:>12,.0f}')

# Convert to DataFrame
df_results = pd.DataFrame(results)

# Sort by MAPE within each metric
df_results = df_results.sort_values(['Metric', 'MAPE (%)']).reset_index(drop=True)

# Save comprehensive results
output_dir = Path('results')
output_dir.mkdir(exist_ok=True)

df_results.to_csv(output_dir / 'forecast_validation_all_models_summary.csv', index=False)
print(f'\n✓ Saved: results/forecast_validation_all_models_summary.csv')

# Create rankings
print('\n4. Creating model rankings...')
print('='*80)

rankings = []
for metric in focus_metrics:
    metric_results = df_results[df_results['Metric'] == metric].copy()
    metric_results['Rank'] = range(1, len(metric_results) + 1)

    print(f'\n{metric.upper()} - Ranked by MAPE:')
    print('-'*80)

    for _, row in metric_results.iterrows():
        rank_emoji = '🥇' if row['Rank'] == 1 else '🥈' if row['Rank'] == 2 else '🥉' if row['Rank'] == 3 else '  '
        print(f"   {rank_emoji} #{row['Rank']} {row['Model']:30s}: {row['MAPE (%)']:6.2f}%")

    rankings.append(metric_results)

df_rankings = pd.concat(rankings, ignore_index=True)
df_rankings.to_csv(output_dir / 'forecast_validation_rankings.csv', index=False)
print(f'\n✓ Saved: results/forecast_validation_rankings.csv')

# Generate monthly detail comparison
print('\n5. Generating monthly detail comparison...')

monthly_details = []

for i, date in enumerate(df_predictions['date']):
    for metric in focus_metrics:
        actual_val = df_actual[metric].iloc[i]

        row_data = {
            'Date': date,
            'Metric': metric,
            'Actual': actual_val
        }

        for model in models:
            col_name = f'{metric}_{model}'
            if col_name in df_predictions.columns:
                pred_val = df_predictions[col_name].iloc[i]
                error = pred_val - actual_val
                error_pct = (error / actual_val) * 100

                row_data[f'{model_names_clean[model]} Pred'] = pred_val
                row_data[f'{model_names_clean[model]} Error'] = error
                row_data[f'{model_names_clean[model]} Error %'] = error_pct

        monthly_details.append(row_data)

df_monthly = pd.DataFrame(monthly_details)
df_monthly.to_csv(output_dir / 'forecast_validation_all_models_monthly.csv', index=False)
print(f'✓ Saved: results/forecast_validation_all_models_monthly.csv')

# Executive Summary
print('\n6. Generating executive summary...')
print('='*80)

print('\n' + '='*80)
print('EXECUTIVE SUMMARY: BEST MODELS')
print('='*80)

for metric in focus_metrics:
    metric_results = df_results[df_results['Metric'] == metric].iloc[0]

    print(f'\n{metric.upper()}:')
    print(f'   Best Model: {metric_results["Model"]}')
    print(f'   MAPE: {metric_results["MAPE (%)"]:6.2f}%')
    print(f'   MAE: {metric_results["MAE"]:>,.0f}')
    print(f'   Cumulative Error (9 months): {metric_results["Cumulative Error"]:>,.0f}')

    # Compare with Human baseline
    human_result = df_results[(df_results['Metric'] == metric) &
                              (df_results['Model'] == 'Human (2024÷12)')].iloc[0]

    improvement = ((human_result['MAPE (%)'] - metric_results['MAPE (%)']) /
                   human_result['MAPE (%)'] * 100)

    if improvement > 0:
        print(f'   ✓ {improvement:.1f}% more accurate than Human method')
    else:
        print(f'   ⚠️  Human method {abs(improvement):.1f}% more accurate')

# Model category performance
print('\n' + '='*80)
print('PERFORMANCE BY MODEL CATEGORY')
print('='*80)

categories = {
    'Traditional': ['Human (2024÷12)', 'Seasonal Naive'],
    'ML (Individual)': ['XGBoost', 'CatBoost', 'LightGBM'],
    'ML (Ensemble)': ['Ensemble (Best)', 'Ensemble (Weighted)', 'Ensemble (Hybrid 60/40)']
}

for category, cat_models in categories.items():
    print(f'\n{category}:')
    for metric in focus_metrics:
        metric_results = df_results[df_results['Metric'] == metric]
        cat_results = metric_results[metric_results['Model'].isin(cat_models)]

        if len(cat_results) > 0:
            best_in_cat = cat_results.iloc[0]
            avg_mape = cat_results['MAPE (%)'].mean()

            print(f'   {metric:20s}: Best={best_in_cat["Model"]:25s} ({best_in_cat["MAPE (%)"]:5.2f}%), Avg={avg_mape:5.2f}%')

print('\n' + '='*80)
print('VALIDATION COMPLETE')
print('='*80)
print('\nGenerated outputs:')
print('  1. forecast_validation_all_models_summary.csv - Overall performance metrics')
print('  2. forecast_validation_rankings.csv - Models ranked by MAPE')
print('  3. forecast_validation_all_models_monthly.csv - Month-by-month details')
print('\nRecommendation:')

# Final recommendation
orders_best = df_results[df_results['Metric'] == 'total_orders'].iloc[0]
revenue_best = df_results[df_results['Metric'] == 'revenue_total'].iloc[0]

print(f'  • For ORDER forecasting: Use {orders_best["Model"]} (MAPE: {orders_best["MAPE (%)"]:5.2f}%)')
print(f'  • For REVENUE forecasting: Use {revenue_best["Model"]} (MAPE: {revenue_best["MAPE (%)"]:5.2f}%)')

if orders_best['Model'] != revenue_best['Model']:
    print('\n  ⚡ Hybrid approach recommended: Different models for different metrics')
else:
    print(f'\n  ⚡ Single model works best: {orders_best["Model"]} for both metrics')
