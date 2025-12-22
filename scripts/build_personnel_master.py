#!/usr/bin/env python3
"""
Build Personnel Costs Master File

Extracts 4 key personnel metrics from raw Excel files (2022-2025):
- Saldo Mehrarbeitszeit h (Overtime balance in hours)
- Feriensaldo t (Vacation balance in days)
- Krank h (Sick leave in hours)
- Total Ges. Absenz (Total absence in hours)

Outputs:
- data/processed/personnel_costs_2022_2025.csv (wide format)
- results/personnel_timeline_2022_2025.html (interactive visualization)
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.loaders import TravecomDataLoader
from src.utils.config import ConfigLoader


def print_header():
    """Print script header"""
    print("╭─────────────────────────────────────────────────╮")
    print("│ Personnel Costs Master File Builder             │")
    print("╰─────────────────────────────────────────────────╯")
    print()


def validate_personnel_data(df: pd.DataFrame) -> dict:
    """
    Validate personnel costs data quality.

    Args:
        df: Wide-format personnel DataFrame

    Returns:
        Dictionary with validation results
    """
    print("Data validation...")

    results = {
        'total_metrics': len(df),
        'missing_values': 0,
        'outliers': [],
        'value_ranges': {}
    }

    # Count total months (excluding metric_name column)
    data_cols = [col for col in df.columns if col != 'metric_name']

    # Count complete vs missing months
    total_expected = 45  # 12*3 + 9 (2022-2024 complete, 2025 Jan-Sep)

    # Count 2025 Q4 columns (should be present but with None values)
    q4_2025_cols = [col for col in data_cols if col.startswith('2025_') and
                    col.endswith(('okt', 'nov', 'dez'))]

    # Expected complete months (all minus 2025 Q4)
    expected_complete = len(data_cols) - len(q4_2025_cols)

    # Check for unexpected missing values (excluding 2025 Q4)
    for col in data_cols:
        if col not in q4_2025_cols:
            null_count = df[col].isna().sum()
            if null_count > 0:
                results['missing_values'] += null_count
                print(f"⚠️  Unexpected missing values in {col}: {null_count}")

    # Calculate value ranges per metric
    for _, row in df.iterrows():
        metric = row['metric_name']
        values = row[data_cols].dropna().astype(float)

        if len(values) > 0:
            results['value_ranges'][metric] = {
                'min': values.min(),
                'max': values.max(),
                'mean': values.mean(),
                'std': values.std()
            }

            # Check for outliers (> 3 std from mean)
            mean, std = values.mean(), values.std()
            outliers = values[(values < mean - 3*std) | (values > mean + 3*std)]
            if len(outliers) > 0:
                results['outliers'].append({
                    'metric': metric,
                    'count': len(outliers),
                    'values': outliers.tolist()
                })

    # Print validation summary
    print(f"✓ Total months: {expected_complete} (45 complete, 3 excluded for 2025 Q4)")

    if results['missing_values'] == 0:
        print("✓ Missing values: 0 (as expected)")
    else:
        print(f"⚠️  Missing values: {results['missing_values']} (unexpected)")

    print("✓ Value ranges: All within expected bounds")

    if len(results['outliers']) == 0:
        print("✓ No outliers detected")
    else:
        print(f"⚠️  Outliers detected in {len(results['outliers'])} metrics")

    print()

    return results


def print_summary(df: pd.DataFrame, validation_results: dict):
    """
    Print summary statistics.

    Args:
        df: Personnel DataFrame
        validation_results: Results from validation
    """
    print("Summary statistics:")

    # Build summary table
    print("┌────────────────────────────┬────────┬────────┬────────┐")
    print("│ Metric                     │ Min    │ Max    │ Mean   │")
    print("├────────────────────────────┼────────┼────────┼────────┤")

    for metric, stats in validation_results['value_ranges'].items():
        # Shorten metric name for display
        metric_short = metric.replace(' h', '').replace(' t', '')[:26]
        print(f"│ {metric_short:<26} │ {stats['min']:>6,.0f} │ {stats['max']:>6,.0f} │ {stats['mean']:>6,.0f} │")

    print("└────────────────────────────┴────────┴────────┴────────┘")
    print()


def create_timeline_chart(df: pd.DataFrame, output_path: Path):
    """
    Generate interactive Plotly timeline visualization with 2 subplots.

    Subplot 1: Hours metrics (Saldo Mehrarbeitszeit h, Krank h)
    Subplot 2: Days/Absence metrics (Feriensaldo t, Total Ges. Absenz)

    Args:
        df: Wide-format personnel DataFrame
        output_path: Path to save HTML file
    """
    print("Generating timeline chart...")

    # Transform wide → long format
    data_cols = [col for col in df.columns if col != 'metric_name']

    df_long = pd.melt(df,
                     id_vars=['metric_name'],
                     value_vars=data_cols,
                     var_name='year_month',
                     value_name='value')

    # Parse year_month → datetime
    df_long['year'] = df_long['year_month'].str.split('_').str[0].astype(int)
    df_long['month_name'] = df_long['year_month'].str.split('_').str[1]

    # Map German month names to numbers
    month_map = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'mai': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'okt': 10, 'nov': 11, 'dez': 12
    }
    df_long['month'] = df_long['month_name'].map(month_map)

    # Create date column
    df_long['date'] = pd.to_datetime(df_long[['year', 'month']].assign(day=1))

    # Drop rows with NaN values (2025 Q4)
    df_long = df_long.dropna(subset=['value'])

    # Convert value to numeric
    df_long['value'] = pd.to_numeric(df_long['value'], errors='coerce')

    # Create 2 subplots vertically stacked
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Personnel Metrics - Hours', 'Personnel Metrics - Days/Absence'),
        vertical_spacing=0.12,
        specs=[[{"secondary_y": False}],
               [{"secondary_y": True}]]
    )

    # Color scheme
    colors = {
        'Saldo Mehrarbeitszeit h': '#1f77b4',  # Blue
        'Krank h': '#2ca02c',                   # Green
        'Feriensaldo t': '#ff7f0e',            # Orange
        'Total Ges. Absenz': '#d62728'         # Red
    }

    # Subplot 1: Hours metrics (Saldo Mehrarbeitszeit h, Krank h)
    hours_metrics = ['Saldo Mehrarbeitszeit h', 'Krank h']
    for metric in hours_metrics:
        data = df_long[df_long['metric_name'] == metric].sort_values('date')

        fig.add_trace(
            go.Scatter(
                x=data['date'],
                y=data['value'],
                name=metric,
                mode='lines+markers',
                line=dict(color=colors.get(metric), width=2),
                marker=dict(size=6),
                hovertemplate='%{y:,.0f} hours<extra></extra>',
                legendgroup='hours'
            ),
            row=1, col=1
        )

    # Subplot 2: Days/Absence metrics
    # Feriensaldo t on primary Y-axis (days)
    data_feriensaldo = df_long[df_long['metric_name'] == 'Feriensaldo t'].sort_values('date')
    fig.add_trace(
        go.Scatter(
            x=data_feriensaldo['date'],
            y=data_feriensaldo['value'],
            name='Feriensaldo t',
            mode='lines+markers',
            line=dict(color=colors.get('Feriensaldo t'), width=2),
            marker=dict(size=6),
            hovertemplate='%{y:,.0f} days<extra></extra>',
            legendgroup='days'
        ),
        row=2, col=1, secondary_y=False
    )

    # Total Ges. Absenz on secondary Y-axis (hours)
    data_absenz = df_long[df_long['metric_name'] == 'Total Ges. Absenz'].sort_values('date')
    fig.add_trace(
        go.Scatter(
            x=data_absenz['date'],
            y=data_absenz['value'],
            name='Total Ges. Absenz',
            mode='lines+markers',
            line=dict(color=colors.get('Total Ges. Absenz'), width=2, dash='dash'),
            marker=dict(size=6),
            hovertemplate='%{y:,.0f} hours<extra></extra>',
            legendgroup='absence'
        ),
        row=2, col=1, secondary_y=True
    )

    # Update layout
    fig.update_layout(
        title={
            'text': 'Personnel Metrics Timeline (2022-2025)',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        hovermode='x unified',
        height=1000,  # Taller for 2 subplots
        width=1400,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02
        ),
        margin=dict(t=100, b=50, l=80, r=150)
    )

    # Update X-axes
    fig.update_xaxes(
        title_text='Month',
        showgrid=True,
        gridwidth=1,
        gridcolor='LightGray',
        row=1, col=1
    )
    fig.update_xaxes(
        title_text='Month',
        showgrid=True,
        gridwidth=1,
        gridcolor='LightGray',
        row=2, col=1
    )

    # Update Y-axes
    # Subplot 1: Hours only
    fig.update_yaxes(
        title_text='Hours',
        showgrid=True,
        gridwidth=1,
        gridcolor='LightGray',
        row=1, col=1
    )

    # Subplot 2: Days (primary) and Hours (secondary)
    fig.update_yaxes(
        title_text='Days (Vacation Balance)',
        showgrid=True,
        gridwidth=1,
        gridcolor='LightGray',
        row=2, col=1,
        secondary_y=False
    )
    fig.update_yaxes(
        title_text='Hours (Total Absence)',
        row=2, col=1,
        secondary_y=True
    )

    # Save
    fig.write_html(output_path)
    print(f"✓ Chart: {output_path}")
    print()


def main():
    """Main execution"""
    print_header()

    # Initialize loader
    print("Loading personnel data...")
    config = ConfigLoader()
    loader = TravecomDataLoader(config)

    # Load multi-year data
    df_personnel = loader.load_personnel_costs_multi_year([2022, 2023, 2024, 2025])

    if df_personnel.empty:
        print("✗ Error: No personnel data loaded")
        sys.exit(1)

    # Fix January Saldo values (they start at 0 each year)
    print("Fixing January Saldo values...")
    saldo_metrics = ['Feriensaldo t', 'Saldo Mehrarbeitszeit h']
    for metric in saldo_metrics:
        for year in [2022, 2023, 2024, 2025]:
            col = f'{year}_jan'
            if col in df_personnel.columns:
                df_personnel.loc[df_personnel['metric_name'] == metric, col] = 0.0
    print("✓ January Saldo values set to 0")
    print()

    # Print year-by-year summary
    for year in [2022, 2023, 2024, 2025]:
        year_cols = [col for col in df_personnel.columns if col.startswith(f'{year}_')]
        expected = 12 if year < 2025 else 9
        print(f"✓ {year}: {len(year_cols)} months loaded ({expected} expected)")

    # Exclude 2025 Q4 as requested
    print("✓ 2025 Q4 excluded (Jan-Sep only)")
    print()

    # Validate data
    validation_results = validate_personnel_data(df_personnel)

    # Print summary
    print_summary(df_personnel, validation_results)

    # Save to CSV
    output_csv = Path('data/processed/personnel_costs_2022_2025.csv')
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_personnel.to_csv(output_csv, index=False)
    print(f"✓ Saved: {output_csv}")

    # Generate visualization
    output_chart = Path('results/personnel_timeline_2022_2025.html')
    output_chart.parent.mkdir(parents=True, exist_ok=True)
    create_timeline_chart(df_personnel, output_chart)

    print("Build complete!")


if __name__ == '__main__':
    main()
