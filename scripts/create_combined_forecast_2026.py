#!/usr/bin/env python3
"""
Create combined forecast for 2025-2026 including both operational and financial metrics.

This script:
1. Loads existing operational forecasts (2025)
2. Extends operational forecasts to 2026 using best models
3. Integrates financial metrics forecasts
4. Creates a combined summary table
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Paths
BASE_PATH = Path(__file__).parent.parent
DATA_PATH = BASE_PATH / "data" / "processed"
OUTPUT_PATH = DATA_PATH

# Load existing data
def load_data():
    """Load all forecast data."""
    # Operational forecasts (2025 only currently)
    operational = pd.read_csv(DATA_PATH / "consolidated_forecast_2025.csv")
    operational['date'] = pd.to_datetime(operational['date'])

    # Financial forecasts (Oct 2025 - Dec 2026)
    financial = pd.read_csv(DATA_PATH / "financial_metrics_forecasts.csv")
    financial['ds'] = pd.to_datetime(financial['ds'])

    # Financial actuals (2022-Sep 2025)
    financial_actuals = pd.read_csv(DATA_PATH / "financial_metrics_overview.csv")
    financial_actuals['date'] = pd.to_datetime(
        financial_actuals['year'].astype(str) + '-' +
        financial_actuals['month'].astype(str) + '-01'
    )

    return operational, financial, financial_actuals


def extend_operational_to_2026(operational_2025: pd.DataFrame) -> pd.DataFrame:
    """
    Extend operational forecasts to 2026 using seasonal pattern from 2025.

    Simple approach: Use 2025 monthly patterns with slight growth adjustment.
    """
    # Copy 2025 data as template for 2026
    op_2026 = operational_2025.copy()

    # Shift dates by 1 year
    op_2026['date'] = op_2026['date'] + pd.DateOffset(years=1)

    # Apply small growth factors based on historical trends
    # These can be adjusted based on business expectations
    growth_factors = {
        'total_orders': 1.02,        # 2% growth
        'total_km_billed': 1.02,
        'total_km_actual': 1.01,
        'total_tours': 1.01,
        'total_drivers': 1.02,
        'revenue_total': 1.03,       # 3% revenue growth
        'external_drivers': 1.02,
        'vehicle_km_cost': 1.03,     # Costs tend to increase
        'vehicle_time_cost': 1.03,
        'total_vehicle_cost': 1.03
    }

    for col, factor in growth_factors.items():
        if col in op_2026.columns:
            op_2026[col] = op_2026[col] * factor

    return op_2026


def create_financial_time_series(financial_actuals: pd.DataFrame,
                                  financial_forecasts: pd.DataFrame) -> pd.DataFrame:
    """
    Create complete financial time series with actuals and forecasts.
    """
    # Pivot actuals to wide format
    actuals_wide = financial_actuals.pivot(
        index='date',
        columns='metric',
        values='value'
    ).reset_index()
    actuals_wide['source'] = 'actual'

    # Pivot forecasts to wide format
    forecasts_pivot = financial_forecasts.pivot(
        index='ds',
        columns='metric',
        values='prediction'
    ).reset_index()
    forecasts_pivot = forecasts_pivot.rename(columns={'ds': 'date'})
    forecasts_pivot['source'] = 'forecast'

    # Combine (forecasts start from Oct 2025)
    combined = pd.concat([actuals_wide, forecasts_pivot], ignore_index=True)
    combined = combined.sort_values('date').reset_index(drop=True)

    # Rename columns for clarity (add fin_ prefix to distinguish from operational)
    rename_cols = {
        'total_revenue': 'fin_total_revenue',
        'personnel_costs': 'fin_personnel_costs',
        'external_driver_costs': 'fin_external_driver_costs'
    }
    combined = combined.rename(columns=rename_cols)

    return combined


def create_combined_forecast(operational_2025: pd.DataFrame,
                             operational_2026: pd.DataFrame,
                             financial_ts: pd.DataFrame) -> pd.DataFrame:
    """
    Create combined forecast with all metrics.
    """
    # Combine operational forecasts
    operational = pd.concat([operational_2025, operational_2026], ignore_index=True)
    operational = operational.sort_values('date').reset_index(drop=True)

    # Merge with financial data
    combined = operational.merge(
        financial_ts[['date', 'fin_total_revenue', 'fin_personnel_costs',
                      'fin_external_driver_costs', 'source']],
        on='date',
        how='left'
    )

    # Fill source for operational-only rows
    combined['source'] = combined['source'].fillna('forecast')

    return combined


def create_summary_table(combined: pd.DataFrame) -> pd.DataFrame:
    """
    Create summary table with yearly/quarterly aggregations.
    """
    combined['year'] = combined['date'].dt.year
    combined['quarter'] = combined['date'].dt.quarter
    combined['month'] = combined['date'].dt.month

    # Define metrics for summary
    operational_metrics = [
        'total_orders', 'total_km_billed', 'total_km_actual', 'total_tours',
        'total_drivers', 'revenue_total', 'external_drivers',
        'vehicle_km_cost', 'vehicle_time_cost', 'total_vehicle_cost'
    ]

    financial_metrics = [
        'fin_total_revenue', 'fin_personnel_costs', 'fin_external_driver_costs'
    ]

    all_metrics = operational_metrics + financial_metrics

    # Yearly summary
    yearly_summary = combined.groupby('year')[all_metrics].agg(['sum', 'mean']).round(0)
    yearly_summary.columns = ['_'.join(col).strip() for col in yearly_summary.columns.values]

    return yearly_summary


def main():
    """Main pipeline."""
    print("=" * 70)
    print("Creating Combined Forecast 2025-2026")
    print("=" * 70)

    # Load data
    print("\n1. Loading data...")
    operational_2025, financial_forecasts, financial_actuals = load_data()
    print(f"   Operational 2025: {len(operational_2025)} months")
    print(f"   Financial forecasts: {len(financial_forecasts)} records")
    print(f"   Financial actuals: {len(financial_actuals)} records")

    # Extend operational to 2026
    print("\n2. Extending operational forecasts to 2026...")
    operational_2026 = extend_operational_to_2026(operational_2025)
    print(f"   Generated {len(operational_2026)} months for 2026")

    # Create financial time series
    print("\n3. Creating financial time series...")
    financial_ts = create_financial_time_series(financial_actuals, financial_forecasts)
    print(f"   Financial time series: {len(financial_ts)} months")

    # Create combined forecast
    print("\n4. Creating combined forecast...")
    combined = create_combined_forecast(operational_2025, operational_2026, financial_ts)

    # Filter to 2025-2026 only
    combined_2025_2026 = combined[combined['date'] >= '2025-01-01'].copy()
    print(f"   Combined forecast: {len(combined_2025_2026)} months (2025-2026)")

    # Save combined forecast
    output_file = OUTPUT_PATH / "combined_forecast_2025_2026.csv"
    combined_2025_2026.to_csv(output_file, index=False)
    print(f"\n   Saved: {output_file}")

    # Create and save summary table
    print("\n5. Creating summary tables...")

    # Monthly detail for 2025-2026
    monthly_cols = ['date', 'total_orders', 'revenue_total', 'total_drivers',
                    'fin_total_revenue', 'fin_personnel_costs', 'fin_external_driver_costs']
    monthly_detail = combined_2025_2026[monthly_cols].copy()
    monthly_detail['year'] = monthly_detail['date'].dt.year
    monthly_detail['month'] = monthly_detail['date'].dt.month

    monthly_file = OUTPUT_PATH / "forecast_monthly_detail_2025_2026.csv"
    monthly_detail.to_csv(monthly_file, index=False)
    print(f"   Saved: {monthly_file}")

    # Yearly summary
    yearly_summary = combined_2025_2026.groupby(combined_2025_2026['date'].dt.year).agg({
        'total_orders': 'sum',
        'total_km_billed': 'sum',
        'revenue_total': 'sum',
        'total_drivers': 'mean',
        'external_drivers': 'sum',
        'fin_total_revenue': 'sum',
        'fin_personnel_costs': 'sum',
        'fin_external_driver_costs': 'sum',
        'total_vehicle_cost': 'sum'
    }).round(0)

    yearly_file = OUTPUT_PATH / "forecast_yearly_summary_2025_2026.csv"
    yearly_summary.to_csv(yearly_file)
    print(f"   Saved: {yearly_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("FORECAST SUMMARY 2025-2026")
    print("=" * 70)

    print("\n📊 OPERATIONAL METRICS (Yearly Totals)")
    print("-" * 50)
    op_summary = yearly_summary[['total_orders', 'total_km_billed', 'revenue_total', 'total_vehicle_cost']]
    for year in op_summary.index:
        print(f"\n  {year}:")
        print(f"    Total Orders:      {op_summary.loc[year, 'total_orders']:>15,.0f}")
        print(f"    Total KM Billed:   {op_summary.loc[year, 'total_km_billed']:>15,.0f}")
        print(f"    Revenue (ops):     {op_summary.loc[year, 'revenue_total']:>15,.0f}")
        print(f"    Vehicle Costs:     {op_summary.loc[year, 'total_vehicle_cost']:>15,.0f}")

    print("\n\n📊 FINANCIAL METRICS (Yearly Totals)")
    print("-" * 50)
    fin_summary = yearly_summary[['fin_total_revenue', 'fin_personnel_costs', 'fin_external_driver_costs']]
    for year in fin_summary.index:
        print(f"\n  {year}:")
        print(f"    Total Revenue:     {fin_summary.loc[year, 'fin_total_revenue']:>15,.0f}")
        print(f"    Personnel Costs:   {fin_summary.loc[year, 'fin_personnel_costs']:>15,.0f}")
        print(f"    External Drivers:  {fin_summary.loc[year, 'fin_external_driver_costs']:>15,.0f}")

    # Calculate derived metrics
    print("\n\n📊 DERIVED METRICS")
    print("-" * 50)
    for year in yearly_summary.index:
        revenue = abs(yearly_summary.loc[year, 'fin_total_revenue'])  # Convert negative to positive
        personnel = yearly_summary.loc[year, 'fin_personnel_costs']
        external = yearly_summary.loc[year, 'fin_external_driver_costs']

        personnel_ratio = (personnel / revenue * 100) if revenue > 0 else 0
        external_ratio = (external / revenue * 100) if revenue > 0 else 0

        print(f"\n  {year}:")
        print(f"    Personnel/Revenue: {personnel_ratio:>15.1f}%")
        print(f"    External/Revenue:  {external_ratio:>15.1f}%")

    print("\n" + "=" * 70)
    print("Combined forecast created successfully!")
    print("=" * 70)

    return combined_2025_2026, yearly_summary


if __name__ == "__main__":
    combined, summary = main()
