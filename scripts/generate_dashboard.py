#!/usr/bin/env python3
"""
Dashboard Generator for Traveco Forecasting System
===================================================

Generates an interactive HTML dashboard from available forecast data.
Uses Plotly for visualizations with Traveco corporate styling.

Usage:
    python scripts/generate_dashboard.py

Output:
    results/forecast_dashboard.html
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import re

# Project root
PROJECT_ROOT = Path(__file__).parent.parent


def load_operational_data():
    """Load operational metrics from validation results"""

    # Load 2025 validation data (actuals + forecasts)
    validation_file = PROJECT_ROOT / 'results' / 'forecast_validation_monthly_detail.csv'
    forecast_file = PROJECT_ROOT / 'results' / 'monthly_forecast_2025_table.csv'

    data = {}

    if validation_file.exists():
        df_val = pd.read_csv(validation_file)
        df_val['date'] = pd.to_datetime(df_val['date'])

        # Extract actuals (Jan-Sep 2025 or available months)
        data['orders_actuals'] = df_val[['date', 'orders_actual']].rename(
            columns={'orders_actual': 'value'}
        )
        data['revenue_actuals'] = df_val[['date', 'revenue_actual']].rename(
            columns={'revenue_actual': 'value'}
        )

        # Best machine predictions (for forecasts)
        data['orders_machine'] = df_val[['date', 'orders_machine']].rename(
            columns={'orders_machine': 'value'}
        )
        data['revenue_machine'] = df_val[['date', 'revenue_machine']].rename(
            columns={'revenue_machine': 'value'}
        )

    if forecast_file.exists():
        df_fc = pd.read_csv(forecast_file)
        df_fc['date'] = pd.to_datetime(df_fc['date'])
        data['forecasts_2025'] = df_fc

    return data


def load_financial_data():
    """Load financial metrics from raw CSV files (2022-2025)"""
    import csv

    financial_data = []

    for year in [2022, 2023, 2024, 2025]:
        file_path = PROJECT_ROOT / 'data' / 'raw' / str(year) / f'{year} Finanzen' / f'{year}.csv'

        if not file_path.exists():
            print(f"  Warning: {file_path} not found")
            continue

        try:
            # Read raw file lines
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                lines = f.readlines()

            # Skip header rows and process each line
            for line_num, line in enumerate(lines[2:], 3):  # Skip header rows
                line = line.strip()
                if not line:
                    continue

                # Use csv reader to properly handle the quoting
                try:
                    reader = csv.reader([line], quotechar='"', doublequote=True)
                    parts = list(reader)[0]
                except:
                    continue

                if len(parts) < 12:
                    continue

                sachkonto = parts[0]

                # Total Betriebsertrag (SK 0140)
                if ',Total Betriebsertrag' in line or sachkonto.startswith('0140'):
                    # This row format: "0140,Total Betriebsertrag,val1,val2,..."
                    # After CSV parsing, parts[0] = "0140,Total Betriebsertrag"
                    # We need to re-split the first part
                    if ',' in sachkonto:
                        actual_parts = sachkonto.split(',')
                        sachkonto = actual_parts[0]
                        # Remaining values follow
                        values = actual_parts[1:] + parts[1:]  # Combine description + values
                    else:
                        values = parts[1:]

                    for i in range(11):  # Jan to Nov
                        idx = i + 1  # Skip description
                        val = parse_german_number(values[idx] if len(values) > idx else None)
                        if val is not None:
                            financial_data.append({
                                'date': f'{year}-{i+1:02d}-01',
                                'metric': 'total_betriebsertrag',
                                'value': abs(val)
                            })

                # Personalaufwand - Lohnaufwand total (SK 0500000)
                elif ',Lohnaufwand' in line or '0500000' in sachkonto:
                    if ',' in sachkonto:
                        actual_parts = sachkonto.split(',')
                        sachkonto = actual_parts[0]
                        values = actual_parts[1:] + parts[1:]
                    else:
                        values = parts[1:]

                    for i in range(11):
                        idx = i + 1
                        val = parse_german_number(values[idx] if len(values) > idx else None)
                        if val is not None:
                            financial_data.append({
                                'date': f'{year}-{i+1:02d}-01',
                                'metric': 'personnel_costs',
                                'value': abs(val)
                            })

                # Transport (SK 35060000)
                elif '35060000' in sachkonto or '35060000' in line:
                    if ',' in sachkonto:
                        actual_parts = sachkonto.split(',')
                        values = actual_parts[1:] + parts[1:]
                    else:
                        values = parts[1:]

                    for i in range(11):
                        idx = i + 1
                        val = parse_german_number(values[idx] if len(values) > idx else None)
                        if val is not None:
                            financial_data.append({
                                'date': f'{year}-{i+1:02d}-01',
                                'metric': 'transport_revenue',
                                'value': abs(val)
                            })

                # Logistik (SK 35070000)
                elif '35070000' in sachkonto or '35070000' in line:
                    if ',' in sachkonto:
                        actual_parts = sachkonto.split(',')
                        values = actual_parts[1:] + parts[1:]
                    else:
                        values = parts[1:]

                    for i in range(11):
                        idx = i + 1
                        val = parse_german_number(values[idx] if len(values) > idx else None)
                        if val is not None:
                            financial_data.append({
                                'date': f'{year}-{i+1:02d}-01',
                                'metric': 'logistics_revenue',
                                'value': abs(val)
                            })

                # Ausgangsfrachten LKW (SK 62800000)
                elif '62800000' in sachkonto or '62800000' in line:
                    if ',' in sachkonto:
                        actual_parts = sachkonto.split(',')
                        values = actual_parts[1:] + parts[1:]
                    else:
                        values = parts[1:]

                    for i in range(11):
                        idx = i + 1
                        val = parse_german_number(values[idx] if len(values) > idx else None)
                        if val is not None:
                            financial_data.append({
                                'date': f'{year}-{i+1:02d}-01',
                                'metric': 'external_driver_costs',
                                'value': abs(val)
                            })

        except Exception as e:
            print(f"  Error loading {file_path}: {e}")
            import traceback
            traceback.print_exc()

    if financial_data:
        df = pd.DataFrame(financial_data)
        df['date'] = pd.to_datetime(df['date'])
        return df

    return None


def parse_german_number(val):
    """Parse German-formatted numbers (1.234,56 or "1,234.56")"""
    if pd.isna(val):
        return None

    val_str = str(val).strip()
    if not val_str or val_str == '-':
        return None

    # Remove quotes
    val_str = val_str.replace('"', '').strip()

    # Handle German format: 1.234,56 -> 1234.56
    if ',' in val_str and '.' in val_str:
        # Check which is decimal separator (last one)
        comma_pos = val_str.rfind(',')
        dot_pos = val_str.rfind('.')

        if comma_pos > dot_pos:
            # German format: 1.234,56
            val_str = val_str.replace('.', '').replace(',', '.')
        else:
            # US format: 1,234.56
            val_str = val_str.replace(',', '')
    elif ',' in val_str:
        # Could be German decimal or US thousands
        parts = val_str.split(',')
        if len(parts) == 2 and len(parts[1]) <= 2:
            # German decimal
            val_str = val_str.replace(',', '.')
        else:
            # US thousands
            val_str = val_str.replace(',', '')

    try:
        return float(val_str)
    except:
        return None


def generate_html_dashboard(op_data, fin_data):
    """Generate the HTML dashboard"""

    # Prepare chart data
    charts_js = []

    # Chart 1: Total Orders
    if 'orders_actuals' in op_data:
        actuals = op_data['orders_actuals']
        forecasts = op_data.get('forecasts_2025')

        # Actuals
        dates_actual = actuals['date'].dt.strftime('%Y-%m-%d').tolist()
        values_actual = actuals['value'].tolist()

        # Forecasts (Dec 2025 onwards from forecast table, or use machine predictions)
        if forecasts is not None:
            fc = forecasts[forecasts['date'] >= '2025-10-01']
            dates_fc = fc['date'].dt.strftime('%Y-%m-%d').tolist()
            values_fc = fc['total_orders'].tolist()
        else:
            dates_fc = []
            values_fc = []

        chart_js = generate_chart_js(
            chart_id='chart_orders',
            title='Aufträge (Orders)',
            dates_actual=dates_actual,
            values_actual=values_actual,
            dates_forecast=dates_fc,
            values_forecast=values_fc,
            value_prefix='',
            value_suffix=''
        )
        charts_js.append(('Aufträge', chart_js, 'chart_orders'))

    # Chart 2: Revenue
    if 'revenue_actuals' in op_data:
        actuals = op_data['revenue_actuals']
        forecasts = op_data.get('forecasts_2025')

        dates_actual = actuals['date'].dt.strftime('%Y-%m-%d').tolist()
        values_actual = actuals['value'].tolist()

        if forecasts is not None:
            fc = forecasts[forecasts['date'] >= '2025-10-01']
            dates_fc = fc['date'].dt.strftime('%Y-%m-%d').tolist()
            values_fc = fc['revenue_total'].tolist()
        else:
            dates_fc = []
            values_fc = []

        chart_js = generate_chart_js(
            chart_id='chart_revenue',
            title='Umsatz (Revenue)',
            dates_actual=dates_actual,
            values_actual=values_actual,
            dates_forecast=dates_fc,
            values_forecast=values_fc,
            value_prefix='CHF ',
            value_suffix=''
        )
        charts_js.append(('Umsatz', chart_js, 'chart_revenue'))

    # Financial charts
    if fin_data is not None:
        # Total Betriebsertrag
        df_metric = fin_data[fin_data['metric'] == 'total_betriebsertrag'].sort_values('date')
        if len(df_metric) > 0:
            # Split into actuals (before Dec 2025) and forecast
            cutoff = pd.Timestamp('2025-11-01')
            actuals = df_metric[df_metric['date'] <= cutoff]

            chart_js = generate_chart_js(
                chart_id='chart_betriebsertrag',
                title='Total Betriebsertrag (SK 0140)',
                dates_actual=actuals['date'].dt.strftime('%Y-%m-%d').tolist(),
                values_actual=actuals['value'].tolist(),
                dates_forecast=[],
                values_forecast=[],
                value_prefix='CHF ',
                value_suffix=''
            )
            charts_js.append(('Betriebsertrag', chart_js, 'chart_betriebsertrag'))

        # Personnel Costs
        df_metric = fin_data[fin_data['metric'] == 'personnel_costs'].sort_values('date')
        if len(df_metric) > 0:
            cutoff = pd.Timestamp('2025-11-01')
            actuals = df_metric[df_metric['date'] <= cutoff]

            chart_js = generate_chart_js(
                chart_id='chart_personnel',
                title='Personalaufwand',
                dates_actual=actuals['date'].dt.strftime('%Y-%m-%d').tolist(),
                values_actual=actuals['value'].tolist(),
                dates_forecast=[],
                values_forecast=[],
                value_prefix='CHF ',
                value_suffix=''
            )
            charts_js.append(('Personalaufwand', chart_js, 'chart_personnel'))

        # Transport + Logistics Revenue
        df_transport = fin_data[fin_data['metric'] == 'transport_revenue'].sort_values('date')
        df_logistics = fin_data[fin_data['metric'] == 'logistics_revenue'].sort_values('date')

        if len(df_transport) > 0 and len(df_logistics) > 0:
            # Combine transport + logistics
            df_combined = df_transport.merge(df_logistics, on='date', suffixes=('_t', '_l'))
            df_combined['value'] = df_combined['value_t'] + df_combined['value_l']

            cutoff = pd.Timestamp('2025-11-01')
            actuals = df_combined[df_combined['date'] <= cutoff]

            chart_js = generate_chart_js(
                chart_id='chart_transport_logistics',
                title='Transport + Logistik (SK 3506+3507)',
                dates_actual=actuals['date'].dt.strftime('%Y-%m-%d').tolist(),
                values_actual=actuals['value'].tolist(),
                dates_forecast=[],
                values_forecast=[],
                value_prefix='CHF ',
                value_suffix=''
            )
            charts_js.append(('Transport+Logistik', chart_js, 'chart_transport_logistics'))

        # External Driver Costs
        df_metric = fin_data[fin_data['metric'] == 'external_driver_costs'].sort_values('date')
        if len(df_metric) > 0:
            cutoff = pd.Timestamp('2025-11-01')
            actuals = df_metric[df_metric['date'] <= cutoff]

            chart_js = generate_chart_js(
                chart_id='chart_external_costs',
                title='Ausgangsfrachten LKW (SK 6280)',
                dates_actual=actuals['date'].dt.strftime('%Y-%m-%d').tolist(),
                values_actual=actuals['value'].tolist(),
                dates_forecast=[],
                values_forecast=[],
                value_prefix='CHF ',
                value_suffix=''
            )
            charts_js.append(('Ausgangsfrachten', chart_js, 'chart_external_costs'))

    # Generate HTML
    html = generate_html_template(charts_js)

    return html


def generate_chart_js(chart_id, title, dates_actual, values_actual,
                       dates_forecast, values_forecast, value_prefix='', value_suffix=''):
    """Generate Plotly.newPlot JavaScript for a chart"""

    # Combine last actual with forecasts for continuity
    if dates_forecast and dates_actual:
        dates_forecast_full = [dates_actual[-1]] + dates_forecast
        values_forecast_full = [values_actual[-1]] + values_forecast
    else:
        dates_forecast_full = dates_forecast
        values_forecast_full = values_forecast

    trace_actual = {
        'x': dates_actual,
        'y': values_actual,
        'type': 'scatter',
        'mode': 'lines+markers',
        'name': 'Ist-Werte',
        'line': {'color': 'rgb(0, 133, 42)', 'width': 2},
        'marker': {'size': 4},
        'hovertemplate': f'{value_prefix}%{{y:,.0f}}{value_suffix}<extra>Ist-Werte</extra>'
    }

    traces = [trace_actual]

    if dates_forecast_full:
        trace_forecast = {
            'x': dates_forecast_full,
            'y': values_forecast_full,
            'type': 'scatter',
            'mode': 'lines+markers',
            'name': 'Prognose',
            'line': {'color': '#dc2626', 'width': 2, 'dash': 'dash'},
            'marker': {'size': 4},
            'hovertemplate': f'{value_prefix}%{{y:,.0f}}{value_suffix}<extra>Prognose</extra>'
        }
        traces.append(trace_forecast)

    layout = {
        'title': {'text': title, 'font': {'size': 14, 'color': '#191919'}},
        'xaxis': {'title': '', 'tickformat': '%b %Y'},
        'yaxis': {'title': '', 'tickformat': ',.0f'},
        'legend': {'orientation': 'h', 'y': -0.15},
        'margin': {'l': 60, 'r': 20, 't': 40, 'b': 60},
        'hovermode': 'x unified',
        'paper_bgcolor': 'white',
        'plot_bgcolor': 'white'
    }

    config = {'responsive': True, 'displayModeBar': False}

    js = f"""Plotly.newPlot('{chart_id}', {json.dumps(traces)}, {json.dumps(layout)}, {json.dumps(config)});"""

    return js


def generate_html_template(charts_js):
    """Generate the complete HTML dashboard"""

    # Separate operational and financial charts
    op_charts = [c for c in charts_js if c[0] in ['Aufträge', 'Umsatz']]
    fin_charts = [c for c in charts_js if c[0] not in ['Aufträge', 'Umsatz']]

    # Generate chart divs
    op_chart_divs = '\n'.join([
        f'            <div class="chart-container"><div id="{c[2]}" style="height: 300px;"></div></div>'
        for c in op_charts
    ])

    fin_chart_divs = '\n'.join([
        f'            <div class="chart-container"><div id="{c[2]}" style="height: 300px;"></div></div>'
        for c in fin_charts
    ])

    # Generate chart scripts
    chart_scripts = '\n        '.join([c[1] for c in charts_js])

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')

    html = f'''<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Traveco Prognose Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        :root {{
            --traveco-green: rgb(0, 133, 42);
            --traveco-green-light: rgba(0, 133, 42, 0.1);
            --traveco-green-dark: rgb(0, 100, 32);
            --traveco-text: #191919;
            --traveco-text-light: #666666;
            --traveco-bg: #f5f5f5;
            --traveco-white: #ffffff;
        }}
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background-color: var(--traveco-bg);
            color: var(--traveco-text);
            padding: 20px;
        }}
        .header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 20px;
            padding: 20px 30px;
            background: var(--traveco-white);
            border-bottom: 3px solid var(--traveco-green);
        }}
        .header h1 {{
            font-size: 1.8rem;
            font-style: italic;
            font-weight: bold;
            margin-bottom: 8px;
        }}
        .header p {{
            font-size: 0.95rem;
            color: var(--traveco-text-light);
        }}
        .legend-info {{
            text-align: center;
            margin-bottom: 15px;
            padding: 12px;
            background: var(--traveco-white);
            border-left: 4px solid var(--traveco-green);
        }}
        .legend-info span {{
            margin: 0 12px;
            font-size: 0.85rem;
        }}
        .legend-actual {{
            color: rgb(0, 133, 42);
            font-weight: 600;
        }}
        .legend-forecast {{
            color: #dc2626;
            font-weight: 600;
        }}
        .section-title {{
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--traveco-text);
            margin: 20px 0 10px 0;
            padding-left: 10px;
            border-left: 4px solid var(--traveco-green);
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }}
        .chart-container {{
            background: var(--traveco-white);
            padding: 15px;
            border: 1px solid #e0e0e0;
        }}
        .summary-table {{
            max-width: 900px;
            margin: 30px auto;
            background: var(--traveco-white);
            padding: 20px;
            border: 1px solid #e0e0e0;
        }}
        .summary-table h3 {{
            font-size: 1rem;
            margin-bottom: 15px;
            padding-left: 10px;
            border-left: 4px solid var(--traveco-green);
        }}
        .summary-table table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }}
        .summary-table th {{
            background: var(--traveco-green);
            color: white;
            padding: 10px;
            text-align: left;
        }}
        .summary-table td {{
            padding: 10px;
            border-bottom: 1px solid #e0e0e0;
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            padding: 20px;
            background: var(--traveco-white);
            font-size: 0.85rem;
            border-top: 3px solid var(--traveco-green);
        }}
        @media (max-width: 768px) {{
            .grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>Prognose Dashboard</h1>
            <p>Generiert: {timestamp}</p>
        </div>
    </div>

    <div class="legend-info">
        <span class="legend-actual">━━ Ist-Werte (Actuals)</span>
        <span class="legend-forecast">┈┈ Prognose (Forecast)</span>
    </div>

    <div class="section-title">Operative Kennzahlen (aus Auftragsdaten)</div>
    <div class="grid">
{op_chart_divs}
    </div>

    <div class="section-title">Finanzkennzahlen (aus Buchhaltung)</div>
    <div class="grid">
{fin_chart_divs}
    </div>

    <div class="summary-table">
        <h3>Modell-Performance Übersicht</h3>
        <table>
            <tr>
                <th>Kennzahl</th>
                <th>Datenquelle</th>
                <th>Bestes Modell</th>
                <th>MAPE</th>
            </tr>
            <tr>
                <td>Aufträge</td>
                <td>QS Auftragsanalyse</td>
                <td>XGBoost</td>
                <td>~4%</td>
            </tr>
            <tr>
                <td>Umsatz (Operativ)</td>
                <td>QS Auftragsanalyse</td>
                <td>Seasonal Naive</td>
                <td>~4%</td>
            </tr>
            <tr>
                <td>Total Betriebsertrag</td>
                <td>Finanzen (SK 0140)</td>
                <td>XGBoost</td>
                <td>~3.5%</td>
            </tr>
            <tr>
                <td>Personalaufwand</td>
                <td>Finanzen</td>
                <td>Prophet</td>
                <td>~3.2%</td>
            </tr>
        </table>
    </div>

    <div class="footer">
        <p>Traveco Forecasting System | Daten bis November 2025</p>
    </div>

    <script>
        {chart_scripts}
    </script>
</body>
</html>'''

    return html


def main():
    """Main entry point"""
    print("=" * 60)
    print("TRAVECO DASHBOARD GENERATOR")
    print("=" * 60)

    print("\n[1/4] Loading operational data...")
    op_data = load_operational_data()
    print(f"  Found: {list(op_data.keys())}")

    print("\n[2/4] Loading financial data...")
    fin_data = load_financial_data()
    if fin_data is not None:
        print(f"  Found {len(fin_data)} financial records")
        print(f"  Metrics: {fin_data['metric'].unique().tolist()}")
    else:
        print("  No financial data found")

    print("\n[3/4] Generating HTML dashboard...")
    html = generate_html_dashboard(op_data, fin_data)

    print("\n[4/4] Saving dashboard...")
    output_file = PROJECT_ROOT / 'results' / 'forecast_dashboard.html'
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\n{'=' * 60}")
    print("DASHBOARD GENERATED SUCCESSFULLY")
    print(f"{'=' * 60}")
    print(f"\nOutput: {output_file}")
    print(f"Size: {len(html):,} bytes")


if __name__ == '__main__':
    main()
