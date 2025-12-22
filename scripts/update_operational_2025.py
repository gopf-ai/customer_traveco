"""
Update operational data with Oct/Nov 2025 order data.

This script:
1. Loads October and November 2025 order files
2. Aggregates to monthly level
3. Updates the combined_forecast_2025_2026.csv with new actual data
4. Preserves forecast data for Dec 2025 onwards
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Paths
DATA_RAW = Path(__file__).parent.parent / "data" / "raw" / "2025"
DATA_PROCESSED = Path(__file__).parent.parent / "data" / "processed"

# Months to process
MONTHS_TO_ADD = [
    ("2025 10 Okt QS Auftragsanalyse.xlsx", 2025, 10),
    ("2025 11 Nov QS Auftragsanalyse.xlsx", 2025, 11),
]


def load_and_aggregate_month(file_path: Path, year: int, month: int) -> dict:
    """Load an order file and aggregate to monthly metrics."""
    print(f"  Loading {file_path.name}...")

    df = pd.read_excel(file_path)
    print(f"    Loaded {len(df):,} records")

    # Count total orders
    total_orders = len(df)

    # Sum revenue (∑ Einnahmen)
    if '∑ Einnahmen' in df.columns:
        df['∑ Einnahmen'] = pd.to_numeric(df['∑ Einnahmen'], errors='coerce')
        revenue_total = df['∑ Einnahmen'].sum()
    else:
        revenue_total = 0
        print("    ⚠️  Revenue column not found")

    # Sum billed KM (Distanz_BE.Auftrag)
    if 'Distanz_BE.Auftrag' in df.columns:
        df['Distanz_BE.Auftrag'] = pd.to_numeric(df['Distanz_BE.Auftrag'], errors='coerce')
        total_km_billed = df['Distanz_BE.Auftrag'].sum()
    else:
        total_km_billed = 0
        print("    ⚠️  Distance column not found")

    # Count external drivers (Nummer.Spedition >= 9000)
    if 'Nummer.Spedition' in df.columns:
        df['Nummer.Spedition'] = pd.to_numeric(df['Nummer.Spedition'], errors='coerce')
        external_drivers = (df['Nummer.Spedition'] >= 9000).sum()
        internal_drivers = ((df['Nummer.Spedition'] > 0) & (df['Nummer.Spedition'] < 9000)).sum()
        total_drivers = external_drivers + internal_drivers
    else:
        external_drivers = 0
        internal_drivers = 0
        total_drivers = 0
        print("    ⚠️  Carrier column not found")

    result = {
        'date': pd.Timestamp(year=year, month=month, day=1),
        'total_orders': total_orders,
        'revenue_total': revenue_total,
        'total_km_billed': total_km_billed,
        'external_drivers': external_drivers,
        'total_drivers': total_drivers,
        'source': 'actual'
    }

    print(f"    ✓ Orders: {total_orders:,}, Revenue: {revenue_total:,.0f}, KM: {total_km_billed:,.0f}")

    return result


def main():
    print("=" * 60)
    print("Update Operational Data with Oct/Nov 2025")
    print("=" * 60)

    # Load existing combined forecast
    combined_path = DATA_PROCESSED / "combined_forecast_2025_2026.csv"
    print(f"\n1. Loading existing data from {combined_path.name}...")
    df_combined = pd.read_csv(combined_path)
    df_combined['date'] = pd.to_datetime(df_combined['date'])

    print(f"   Existing records: {len(df_combined)}")
    print(f"   Date range: {df_combined['date'].min()} to {df_combined['date'].max()}")

    # Process Oct and Nov 2025
    print("\n2. Processing new months...")
    new_records = []

    for filename, year, month in MONTHS_TO_ADD:
        file_path = DATA_RAW / filename
        if file_path.exists():
            record = load_and_aggregate_month(file_path, year, month)
            new_records.append(record)
        else:
            print(f"  ⚠️  File not found: {file_path}")

    if not new_records:
        print("\n⚠️  No new data to add!")
        return

    # Create DataFrame from new records
    df_new = pd.DataFrame(new_records)

    # Remove Oct/Nov 2025 from existing data (if they exist as forecasts)
    dates_to_remove = df_new['date'].tolist()
    df_combined = df_combined[~df_combined['date'].isin(dates_to_remove)]

    # Fill missing columns in new data with NaN (for tour-based metrics we don't have)
    for col in df_combined.columns:
        if col not in df_new.columns:
            df_new[col] = np.nan

    # Ensure column order matches
    df_new = df_new[df_combined.columns]

    # Combine
    df_updated = pd.concat([df_combined, df_new], ignore_index=True)
    df_updated = df_updated.sort_values('date').reset_index(drop=True)

    # Summary
    print("\n3. Summary of updated data:")
    df_2025_actual = df_updated[(df_updated['date'].dt.year == 2025) & (df_updated['source'] == 'actual')]
    print(f"   2025 actual months: {sorted(df_2025_actual['date'].dt.month.unique())}")
    print(f"   Total records: {len(df_updated)}")

    # Save
    print(f"\n4. Saving to {combined_path.name}...")
    df_updated.to_csv(combined_path, index=False)
    print("   ✓ Saved!")

    # Also update monthly_aggregated_full_company.csv if it exists
    monthly_path = DATA_PROCESSED / "monthly_aggregated_full_company.csv"
    if monthly_path.exists():
        print(f"\n5. Updating {monthly_path.name}...")
        df_monthly = pd.read_csv(monthly_path)
        df_monthly['date'] = pd.to_datetime(df_monthly['date'])

        # Check if we need to add 2025 data
        max_date = df_monthly['date'].max()
        print(f"   Current max date: {max_date}")

        # Add new records for 2025
        for record in new_records:
            # Create a row matching the monthly format
            new_row = {
                'year_month': record['date'].strftime('%Y-%m'),
                'total_orders': record['total_orders'],
                'external_drivers': record['external_drivers'],
                'internal_drivers': record['total_drivers'] - record['external_drivers'],
                'revenue_total': record['revenue_total'],
                'total_km_billed': record['total_km_billed'],
                'date': record['date'],
                'total_drivers': record['total_drivers'],
                'year': record['date'].year,
                'month': record['date'].month,
            }

            # Check if this date already exists
            if record['date'] not in df_monthly['date'].values:
                df_monthly = pd.concat([df_monthly, pd.DataFrame([new_row])], ignore_index=True)
                print(f"   Added {record['date'].strftime('%Y-%m')}")

        df_monthly = df_monthly.sort_values('date').reset_index(drop=True)
        df_monthly.to_csv(monthly_path, index=False)
        print("   ✓ Saved!")

    print("\n" + "=" * 60)
    print("Update complete! Run create_forecast_visualization.py to regenerate dashboard.")
    print("=" * 60)


if __name__ == "__main__":
    main()
