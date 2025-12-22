#!/usr/bin/env python3
"""
Analyze correlation between working_days and operational metrics.

This script calculates Pearson correlations to determine which operational
metrics should use working_days as a feature (similar to financial pipeline).

Threshold: |r| > 0.3 and p < 0.01 for inclusion.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# Configuration
DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "monthly_aggregated_full_company.csv"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "processed" / "working_days_correlation_analysis.csv"

# Operational metrics to analyze
OPERATIONAL_METRICS = [
    'total_orders',
    'total_km_billed',
    'total_km_actual',
    'total_tours',
    'total_drivers',
    'revenue_total',
    'external_drivers',
    'vehicle_km_cost',
    'vehicle_time_cost',
    'total_vehicle_cost',
]

# Thresholds for inclusion
CORRELATION_THRESHOLD = 0.3
PVALUE_THRESHOLD = 0.01


def main():
    print("=" * 70)
    print("Working Days Correlation Analysis for Operational Metrics")
    print("=" * 70)

    # Load data
    print(f"\nLoading data from {DATA_PATH.name}...")
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    print(f"  Loaded {len(df)} records ({df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')})")

    # Check for working_days column
    if 'working_days' not in df.columns:
        raise ValueError("working_days column not found in data!")

    # Remove rows with missing working_days
    df_clean = df.dropna(subset=['working_days'])
    print(f"  {len(df_clean)} records with valid working_days")

    # Calculate correlations
    print(f"\n{'='*70}")
    print("Correlation Analysis Results")
    print(f"{'='*70}")
    print(f"\n{'Metric':<25} {'Correlation':>12} {'P-Value':>12} {'Significant':>12} {'Include':>10}")
    print("-" * 70)

    results = []
    metrics_to_include = []

    for metric in OPERATIONAL_METRICS:
        if metric not in df_clean.columns:
            print(f"  {metric:<25} {'N/A':>12} {'N/A':>12} {'N/A':>12} {'N/A':>10}")
            continue

        # Get valid pairs (both working_days and metric non-null)
        mask = df_clean[metric].notna()
        working_days = df_clean.loc[mask, 'working_days'].values
        metric_values = df_clean.loc[mask, metric].values

        if len(working_days) < 3:
            print(f"  {metric:<25} {'N/A':>12} {'<3 pts':>12} {'N/A':>12} {'N/A':>10}")
            continue

        # Calculate Pearson correlation
        corr, pvalue = stats.pearsonr(working_days, metric_values)

        # Determine significance and inclusion
        significant = pvalue < PVALUE_THRESHOLD
        include = abs(corr) > CORRELATION_THRESHOLD and significant

        # Format output
        sig_marker = "Yes" if significant else "No"
        include_marker = "YES" if include else "No"

        if include:
            metrics_to_include.append(metric)
            print(f"  {metric:<25} {corr:>12.4f} {pvalue:>12.6f} {sig_marker:>12} {include_marker:>10} ***")
        else:
            print(f"  {metric:<25} {corr:>12.4f} {pvalue:>12.6f} {sig_marker:>12} {include_marker:>10}")

        results.append({
            'metric': metric,
            'correlation': corr,
            'p_value': pvalue,
            'significant': significant,
            'include_working_days': include,
            'n_observations': len(working_days)
        })

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_PATH, index=False)
    print(f"\n\nResults saved to: {OUTPUT_PATH}")

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\nThresholds: |correlation| > {CORRELATION_THRESHOLD}, p-value < {PVALUE_THRESHOLD}")
    print(f"\nMetrics that should use working_days feature:")

    if metrics_to_include:
        print("\nMETRICS_WITH_WORKING_DAYS = [")
        for m in metrics_to_include:
            corr_val = df_results[df_results['metric'] == m]['correlation'].values[0]
            print(f"    '{m}',  # r = {corr_val:.3f}")
        print("]")
    else:
        print("  None - no metrics meet the correlation threshold")

    print(f"\nMetrics to EXCLUDE from working_days feature:")
    excluded = [m for m in OPERATIONAL_METRICS if m not in metrics_to_include and m in df_clean.columns]
    for m in excluded:
        row = df_results[df_results['metric'] == m]
        if len(row) > 0:
            corr_val = row['correlation'].values[0]
            pval = row['p_value'].values[0]
            print(f"  - {m}: r = {corr_val:.3f}, p = {pval:.4f}")

    # Comparison with financial metrics
    print(f"\n{'='*70}")
    print("COMPARISON WITH FINANCIAL PIPELINE")
    print(f"{'='*70}")
    print("""
Financial Pipeline (reference):
  - total_revenue:       r = -0.572 (included)
  - total_betriebsertrag: r = -0.568 (included)
  - ebt:                 r = -0.503 (included)
  - personnel_costs:     r = -0.251 (excluded)
  - external_driver_costs: r = 0.203 (excluded)
""")

    return df_results, metrics_to_include


if __name__ == "__main__":
    results, included = main()
