#!/usr/bin/env python3
"""
Create static HTML visualization dashboard for Traveco forecasts (German).

Shows 5 key metrics with historical data and forecasts:
- Operational: Aufträge, Umsatz (operativ)
- Financial: Betriebsertrag, Personalaufwand, Ausgangsfrachten LKW
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go


# Paths
BASE_PATH = Path(__file__).parent.parent
DATA_PATH = BASE_PATH / "data" / "processed"
OUTPUT_PATH = BASE_PATH / "results"


# Metric configurations with German labels (Traveco terminology)
METRICS_CONFIG = {
    'total_orders': {
        'title': 'Aufträge',
        'subtitle': 'Monatliches Auftragsvolumen',
        'unit': 'Anzahl',
        'type': 'operational',
        'table_header': 'Aufträge'
    },
    'revenue_total': {
        'title': 'Umsatz (Operativ)',
        'subtitle': 'Umsatz aus Auftragsdaten',
        'unit': 'CHF',
        'type': 'operational',
        'table_header': 'Umsatz (Op.)'
    },
    'total_revenue': {
        'title': 'Betriebsertrag (SK 35060000+35070000)',
        'subtitle': 'Bestes Modell: XGBoost (MAPE: 2.59%)',
        'unit': 'CHF',
        'type': 'financial',
        'invert': True,  # Revenue is negative in accounting
        'table_header': 'Betriebsertrag (SK 3506+3507)'
    },
    'personnel_costs': {
        'title': 'Personalaufwand (SK 0151)',
        'subtitle': 'Bestes Modell: XGBoost+Personal (MAPE: 2.55%)',
        'unit': 'CHF',
        'type': 'financial',
        'table_header': 'Personalaufw. (SK 0151)'
    },
    'external_driver_costs': {
        'title': 'Ausgangsfrachten LKW (SK 62802000)',
        'subtitle': 'Bestes Modell: XGBoost (MAPE: 5.73%)',
        'unit': 'CHF',
        'type': 'financial',
        'table_header': 'Ausg.frachten (SK 6280)'
    }
}


def load_operational_data():
    """Load operational data (historical 2022-2024 + forecasts 2025-2026)."""
    # Historical data (2022-2024)
    hist_df = pd.read_csv(DATA_PATH / "monthly_aggregated_full_company.csv")
    hist_df['date'] = pd.to_datetime(hist_df['date'])
    hist_df = hist_df[hist_df['date'].dt.year <= 2024]
    hist_df['source'] = 'actual'

    # 2025-2026 data (actuals + forecasts)
    forecast_df = pd.read_csv(DATA_PATH / "combined_forecast_2025_2026.csv")
    forecast_df['date'] = pd.to_datetime(forecast_df['date'])

    # Combine
    hist_subset = hist_df[['date', 'total_orders', 'revenue_total', 'source']].copy()
    forecast_subset = forecast_df[['date', 'total_orders', 'revenue_total', 'source']].copy()

    combined = pd.concat([hist_subset, forecast_subset], ignore_index=True)
    combined = combined.sort_values('date').reset_index(drop=True)

    return combined


def load_financial_data():
    """Load financial data (actuals 2022-Sep 2025 + forecasts Oct 2025-Dec 2026)."""
    # Actuals (2022 - Sep 2025)
    actuals_df = pd.read_csv(DATA_PATH / "financial_metrics_overview.csv")
    actuals_df['date'] = pd.to_datetime(
        actuals_df['year'].astype(str) + '-' +
        actuals_df['month'].astype(str).str.zfill(2) + '-01'
    )
    actuals_df['source'] = 'actual'

    # Forecasts (Oct 2025 - Dec 2026)
    forecasts_df = pd.read_csv(DATA_PATH / "financial_metrics_forecasts.csv")
    forecasts_df['date'] = pd.to_datetime(forecasts_df['ds'])
    forecasts_df = forecasts_df.rename(columns={'prediction': 'value'})
    forecasts_df['source'] = 'forecast'

    return actuals_df, forecasts_df


def build_operational_time_series(metric_name, operational_df):
    """Build time series for an operational metric."""
    df = operational_df[['date', metric_name, 'source']].copy()
    df = df.rename(columns={metric_name: 'value'})
    df = df.sort_values('date').reset_index(drop=True)
    return df


def build_financial_time_series(metric_name, actuals_df, forecasts_df):
    """Build time series for a financial metric."""
    actual_data = actuals_df[actuals_df['metric'] == metric_name][['date', 'value', 'source']].copy()
    forecast_data = forecasts_df[forecasts_df['metric'] == metric_name][['date', 'value', 'source']].copy()
    combined = pd.concat([actual_data, forecast_data], ignore_index=True)
    combined = combined.sort_values('date').reset_index(drop=True)
    return combined


def create_metric_chart(metric_name, time_series_df):
    """Create a Plotly figure for a single metric."""
    config = METRICS_CONFIG[metric_name]
    df = time_series_df.copy()

    # Invert if needed (financial revenue is negative)
    if config.get('invert', False):
        df['value'] = -df['value']

    # Split into actual and forecast
    df_actual = df[df['source'] == 'actual'].copy()
    df_forecast = df[df['source'] == 'forecast'].copy()

    # For continuity, include last actual point in forecast line
    if not df_actual.empty and not df_forecast.empty:
        last_actual = df_actual.iloc[-1:].copy()
        last_actual['source'] = 'forecast'
        df_forecast = pd.concat([last_actual, df_forecast], ignore_index=True)
        df_forecast = df_forecast.sort_values('date')

    fig = go.Figure()

    # Convert to lists
    actual_x = df_actual['date'].dt.strftime('%Y-%m-%d').tolist()
    actual_y = df_actual['value'].tolist()

    # Actual line (Traveco green)
    fig.add_trace(go.Scatter(
        x=actual_x,
        y=actual_y,
        mode='lines+markers',
        name='Ist-Werte',
        line=dict(color='rgb(0, 133, 42)', width=2.5),
        marker=dict(size=5),
        hovertemplate='%{x}<br>' + config['unit'] + ': %{y:,.0f}<extra>Ist</extra>'
    ))

    # Forecast line
    if not df_forecast.empty:
        forecast_x = df_forecast['date'].dt.strftime('%Y-%m-%d').tolist()
        forecast_y = df_forecast['value'].tolist()

        fig.add_trace(go.Scatter(
            x=forecast_x,
            y=forecast_y,
            mode='lines+markers',
            name='Prognose',
            line=dict(color='#dc2626', width=3),
            marker=dict(size=8, symbol='diamond'),
            hovertemplate='%{x}<br>' + config['unit'] + ': %{y:,.0f}<extra>Prognose</extra>'
        ))

    # Add vertical line at forecast start (Oct 2025)
    fig.add_shape(
        type='line',
        x0='2025-10-01',
        x1='2025-10-01',
        y0=0,
        y1=1,
        yref='paper',
        line=dict(color='gray', width=1, dash='dot')
    )

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"<b>{config['title']}</b><br><span style='font-size:12px;color:gray'>{config['subtitle']}</span>",
            font=dict(size=14)
        ),
        xaxis_title='',
        yaxis_title=config['unit'],
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        margin=dict(l=80, r=40, t=80, b=50),
        height=400,
        xaxis=dict(
            range=['2022-01-01', '2027-01-01'],
            type='date',
            dtick='M6',
            tickformat='%b %Y'
        )
    )

    fig.update_yaxes(tickformat=',')

    return fig


def build_data_table(operational_df, fin_actuals, fin_forecasts):
    """Build comprehensive data table for all metrics."""
    # Create date range
    dates = pd.date_range('2022-01-01', '2026-12-01', freq='MS')
    table_df = pd.DataFrame({'date': dates})
    table_df['Jahr'] = table_df['date'].dt.year
    table_df['Monat'] = table_df['date'].dt.strftime('%b')

    # Add operational metrics
    for metric in ['total_orders', 'revenue_total']:
        ts = build_operational_time_series(metric, operational_df)
        ts = ts.rename(columns={'value': metric})
        table_df = table_df.merge(ts[['date', metric]], on='date', how='left')

    # Add financial metrics
    for metric in ['total_revenue', 'personnel_costs', 'external_driver_costs']:
        ts = build_financial_time_series(metric, fin_actuals, fin_forecasts)
        # Invert revenue
        if metric == 'total_revenue':
            ts['value'] = -ts['value']
        ts = ts.rename(columns={'value': metric})
        table_df = table_df.merge(ts[['date', metric]], on='date', how='left')

    # Add source column
    table_df['Typ'] = table_df['date'].apply(
        lambda d: 'Prognose' if d >= pd.Timestamp('2025-10-01') else 'Ist'
    )

    return table_df


def generate_yearly_summary_html(table_df):
    """Generate yearly summary table with year-over-year changes."""
    metrics = ['total_orders', 'revenue_total', 'total_revenue', 'personnel_costs', 'external_driver_costs']
    metric_labels = {
        'total_orders': 'Aufträge',
        'revenue_total': 'Umsatz (Op.)',
        'total_revenue': 'Betriebsertrag<br><small>(SK 3506+3507)</small>',
        'personnel_costs': 'Personalaufw.<br><small>(SK 0151)</small>',
        'external_driver_costs': 'Ausg.frachten<br><small>(SK 6280)</small>'
    }

    # Group by year and sum
    yearly = table_df.groupby('Jahr')[metrics].sum().reset_index()

    # Calculate year-over-year changes
    changes = {}
    for metric in metrics:
        changes[metric] = yearly[metric].pct_change() * 100

    html = """
    <div class="yearly-summary">
        <h3>Jahresübersicht mit Veränderung zum Vorjahr</h3>
        <table>
            <thead>
                <tr>
                    <th>Jahr</th>
    """

    for metric in metrics:
        html += f"<th>{metric_labels[metric]}</th>"

    html += """
                </tr>
            </thead>
            <tbody>
    """

    for idx, row in yearly.iterrows():
        year = int(row['Jahr'])
        # Mark 2025/2026 as partially or fully forecast
        year_note = ""
        row_class = ""
        if year == 2025:
            year_note = " *"
            row_class = "partial-forecast"
        elif year == 2026:
            year_note = " **"
            row_class = "full-forecast"

        html += f'<tr class="{row_class}">'
        html += f'<td><strong>{year}{year_note}</strong></td>'

        for metric in metrics:
            val = row[metric]
            change = changes[metric].iloc[idx]

            if pd.isna(val):
                html += '<td class="number">-</td>'
            else:
                # Format value
                if metric == 'total_orders':
                    val_str = f"{val:,.0f}"
                else:
                    val_str = f"{val:,.0f} CHF"

                # Format change
                if pd.isna(change) or idx == 0:
                    change_str = ""
                else:
                    arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
                    color = "#16a34a" if change > 0 else "#dc2626" if change < 0 else "#6b7280"
                    change_str = f'<br><span style="color:{color};font-size:0.8rem">{arrow} {abs(change):.1f}%</span>'

                html += f'<td class="number">{val_str}{change_str}</td>'

        html += '</tr>'

    html += """
            </tbody>
        </table>
        <p class="table-note">
            * 2025: Jan-Sep Ist-Werte, Okt-Dez Prognose<br>
            ** 2026: Vollständig Prognose
        </p>
    </div>
    """

    return html


def generate_table_html(table_df):
    """Generate HTML table from dataframe."""
    # German month names
    month_de = {
        'Jan': 'Jan', 'Feb': 'Feb', 'Mar': 'Mär', 'Apr': 'Apr',
        'May': 'Mai', 'Jun': 'Jun', 'Jul': 'Jul', 'Aug': 'Aug',
        'Sep': 'Sep', 'Oct': 'Okt', 'Nov': 'Nov', 'Dec': 'Dez'
    }
    table_df['Monat'] = table_df['Monat'].map(month_de)

    html = """
    <table class="data-table">
        <thead>
            <tr>
                <th>Jahr</th>
                <th>Monat</th>
                <th>Typ</th>
                <th>Aufträge</th>
                <th>Umsatz (Op.)</th>
                <th>Betriebsertrag<br><small>(SK 3506+3507)</small></th>
                <th>Personalaufw.<br><small>(SK 0151)</small></th>
                <th>Ausg.frachten<br><small>(SK 6280)</small></th>
            </tr>
        </thead>
        <tbody>
    """

    for _, row in table_df.iterrows():
        typ_class = 'forecast-row' if row['Typ'] == 'Prognose' else ''

        def fmt(val):
            if pd.isna(val):
                return '-'
            return f"{val:,.0f}"

        html += f"""
            <tr class="{typ_class}">
                <td>{row['Jahr']}</td>
                <td>{row['Monat']}</td>
                <td><span class="badge {'badge-forecast' if row['Typ'] == 'Prognose' else 'badge-actual'}">{row['Typ']}</span></td>
                <td class="number">{fmt(row['total_orders'])}</td>
                <td class="number">{fmt(row['revenue_total'])}</td>
                <td class="number">{fmt(row['total_revenue'])}</td>
                <td class="number">{fmt(row['personnel_costs'])}</td>
                <td class="number">{fmt(row['external_driver_costs'])}</td>
            </tr>
        """

    html += """
        </tbody>
    </table>
    """

    return html


def create_dashboard():
    """Create the complete HTML dashboard with tabs."""
    print("  Lade operative Daten...")
    operational_df = load_operational_data()

    print("  Lade Finanzdaten...")
    fin_actuals, fin_forecasts = load_financial_data()

    # Build time series for each metric
    time_series = {}

    for metric in ['total_orders', 'revenue_total']:
        print(f"  Erstelle Zeitreihe für: {metric}")
        time_series[metric] = build_operational_time_series(metric, operational_df)

    for metric in ['total_revenue', 'personnel_costs', 'external_driver_costs']:
        print(f"  Erstelle Zeitreihe für: {metric}")
        time_series[metric] = build_financial_time_series(metric, fin_actuals, fin_forecasts)

    # Create figures
    figures = {}
    for metric in METRICS_CONFIG.keys():
        print(f"  Erstelle Diagramm für: {METRICS_CONFIG[metric]['title']}")
        figures[metric] = create_metric_chart(metric, time_series[metric])

    # Build data table
    print("  Erstelle Datentabelle...")
    table_df = build_data_table(operational_df, fin_actuals, fin_forecasts)
    yearly_summary_html = generate_yearly_summary_html(table_df.copy())
    table_html = generate_table_html(table_df)

    # Build HTML content
    html_content = """<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Traveco Prognose Dashboard 2025-2026</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        /* Traveco Corporate Identity */
        @font-face {
            font-family: 'AkkuratLL';
            src: local('AkkuratLL-Regular');
            font-weight: normal;
            font-style: normal;
        }
        @font-face {
            font-family: 'AkkuratLL';
            src: local('AkkuratLL-Bold');
            font-weight: bold;
            font-style: normal;
        }
        @font-face {
            font-family: 'AkkuratLL';
            src: local('AkkuratLL-BoldItalic');
            font-weight: bold;
            font-style: italic;
        }

        :root {
            --traveco-green: rgb(0, 133, 42);
            --traveco-green-light: rgba(0, 133, 42, 0.1);
            --traveco-green-dark: rgb(0, 100, 32);
            --traveco-text: #191919;
            --traveco-text-light: #666666;
            --traveco-bg: #f5f5f5;
            --traveco-white: #ffffff;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        body {
            font-family: 'AkkuratLL', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background-color: var(--traveco-bg);
            color: var(--traveco-text);
            padding: 20px;
        }
        h1, h2, h3 {
            font-family: 'AkkuratLL', -apple-system, BlinkMacSystemFont, sans-serif;
            font-weight: bold;
            font-style: italic;
        }
        .header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 20px;
            padding: 20px 30px;
            background: var(--traveco-white);
            border-radius: 0;
            color: var(--traveco-text);
            border-bottom: 3px solid var(--traveco-green);
        }
        .header-content {
            text-align: left;
        }
        .header h1 {
            font-size: 1.8rem;
            margin-bottom: 8px;
            font-style: italic;
            color: var(--traveco-text);
        }
        .header p {
            font-size: 0.95rem;
            color: var(--traveco-text-light);
        }
        .header-logo {
            height: 40px;
        }
        .header-logo svg {
            height: 100%;
            width: auto;
        }

        /* Tabs */
        .tabs {
            display: flex;
            justify-content: center;
            margin-bottom: 20px;
            gap: 10px;
        }
        .tab-btn {
            font-family: 'AkkuratLL', -apple-system, BlinkMacSystemFont, sans-serif;
            padding: 12px 30px;
            font-size: 1rem;
            font-weight: 600;
            border: 2px solid var(--traveco-green);
            border-radius: 0;
            cursor: pointer;
            transition: all 0.2s;
            background: var(--traveco-white);
            color: var(--traveco-green);
        }
        .tab-btn:hover {
            background: var(--traveco-green-light);
        }
        .tab-btn.active {
            background: var(--traveco-green);
            color: white;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }

        .legend-info {
            text-align: center;
            margin-bottom: 15px;
            padding: 12px;
            background: var(--traveco-white);
            border-radius: 0;
            border-left: 4px solid var(--traveco-green);
        }
        .legend-info span {
            margin: 0 12px;
            font-size: 0.85rem;
        }
        .legend-actual {
            color: rgb(0, 133, 42);
            font-weight: 600;
        }
        .legend-forecast {
            color: #dc2626;
            font-weight: 600;
        }
        .section-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--traveco-text);
            margin: 20px 0 10px 0;
            padding-left: 10px;
            border-left: 4px solid var(--traveco-green);
        }
        .grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 20px;
            max-width: 1200px;
            margin: 0 auto;
        }
        .chart-container {
            background: var(--traveco-white);
            border-radius: 0;
            padding: 12px;
            border: 1px solid #e0e0e0;
        }

        /* Data Table Styles */
        .table-container {
            max-width: 1400px;
            margin: 0 auto;
            background: var(--traveco-white);
            border-radius: 0;
            padding: 20px;
            border: 1px solid #e0e0e0;
            overflow-x: auto;
        }
        .data-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
        }
        .data-table th {
            background: var(--traveco-green);
            color: white;
            padding: 12px 10px;
            text-align: left;
            font-weight: 600;
            border-bottom: 2px solid var(--traveco-green-dark);
            position: sticky;
            top: 0;
        }
        .data-table td {
            padding: 10px;
            border-bottom: 1px solid #e0e0e0;
        }
        .data-table .number {
            text-align: right;
            font-family: 'AkkuratLL', 'SF Mono', Consolas, monospace;
        }
        .data-table tr:hover {
            background: var(--traveco-green-light);
        }
        .data-table .forecast-row {
            background: #fff8e6;
        }
        .data-table .forecast-row:hover {
            background: #ffecb3;
        }
        .badge {
            padding: 3px 8px;
            border-radius: 0;
            font-size: 0.75rem;
            font-weight: 600;
        }
        .badge-actual {
            background: var(--traveco-green-light);
            color: var(--traveco-green-dark);
        }
        .badge-forecast {
            background: #fee2e2;
            color: #dc2626;
        }

        .summary-table {
            max-width: 900px;
            margin: 20px auto;
            background: var(--traveco-white);
            border-radius: 0;
            padding: 15px;
            border: 1px solid #e0e0e0;
        }
        .summary-table h3 {
            font-size: 1rem;
            margin-bottom: 10px;
            color: var(--traveco-text);
        }
        .summary-table table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
        }
        .summary-table th, .summary-table td {
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }
        .summary-table th {
            background: var(--traveco-green);
            color: white;
            font-weight: 600;
        }

        /* Yearly summary table */
        .yearly-summary {
            max-width: 1200px;
            margin: 20px auto;
            background: var(--traveco-white);
            border-radius: 0;
            padding: 20px;
            border: 1px solid #e0e0e0;
        }
        .yearly-summary h3 {
            font-size: 1.1rem;
            margin-bottom: 15px;
            color: var(--traveco-text);
            padding-left: 10px;
            border-left: 4px solid var(--traveco-green);
        }
        .yearly-summary table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }
        .yearly-summary th {
            background: var(--traveco-green);
            color: white;
            padding: 12px 15px;
            text-align: right;
            font-weight: 600;
            border-bottom: 2px solid var(--traveco-green-dark);
        }
        .yearly-summary th:first-child {
            text-align: left;
        }
        .yearly-summary td {
            padding: 15px;
            border-bottom: 1px solid #e0e0e0;
            vertical-align: top;
        }
        .yearly-summary td.number {
            text-align: right;
            font-family: 'AkkuratLL', 'SF Mono', Consolas, monospace;
        }
        .yearly-summary tr:hover {
            background: var(--traveco-green-light);
        }
        .yearly-summary tr.partial-forecast {
            background: #fff8e6;
        }
        .yearly-summary tr.partial-forecast:hover {
            background: #ffecb3;
        }
        .yearly-summary tr.full-forecast {
            background: #ffe0b2;
        }
        .yearly-summary tr.full-forecast:hover {
            background: #ffcc80;
        }
        .yearly-summary .table-note {
            margin-top: 12px;
            font-size: 0.8rem;
            color: var(--traveco-text-light);
            padding-left: 10px;
        }
        .footer {
            text-align: center;
            margin-top: 20px;
            padding: 20px;
            background: var(--traveco-white);
            color: var(--traveco-text);
            font-size: 0.85rem;
            border-top: 3px solid var(--traveco-green);
        }
        .footer a {
            color: var(--traveco-green);
            text-decoration: underline;
        }
        @media (max-width: 768px) {
            .header h1 {
                font-size: 1.3rem;
            }
            .tab-btn {
                padding: 10px 20px;
                font-size: 0.9rem;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="header-content">
            <h1>Prognose Dashboard</h1>
            <p>Historische Daten (Jan 2022 - Sep 2025) | Prognose (Okt 2025 - Dez 2026)</p>
        </div>
        <div class="header-logo">
            <svg xmlns="http://www.w3.org/2000/svg" width="235.791" height="32.909" viewBox="0 0 235.791 32.909">
                <g transform="translate(-155.906 -401.163)">
                    <path d="M228.806,437.707h-8.6l2.261-11.95h11.094Zm41.241-11.95H261.03l-2.26,11.95h11.215Zm-71.987,0H187.32l-2.261,11.95h10.73Zm-29.155,0h-10.74l-2.26,11.949h10.728Zm203.908,11.621c-2.221-2.683-2.042-6.9-1.286-11.622h-8.05l-2.26,11.95h11.905C373.019,437.6,372.91,437.5,372.814,437.379Zm-68.856-11.622h-10.74l-2.261,11.95h10.73Zm34.049,11.622c-2.224-2.686-2.042-6.9-1.286-11.622h-8.045l-2.26,11.95h11.9C338.211,437.6,338.1,437.5,338.007,437.379Z" transform="translate(0 -4.136)" fill="#004152"></path>
                    <path d="M187.355,401.665l-1.007,5.3h-5.569l-5.057,26.607h-5.006l5.057-26.607H170.2l1.007-5.3Zm25.943,16.2-.017.088c1.055.176,4.22.529,3.13,6.263-.386,2.03-1.427,8.029-1.215,9.352h-4.883c-.293-1.941.252-4.1.646-6.176.72-3.794,1.6-7.015-2.274-7.015h-1.3l-2.508,13.191h-5.006l6.065-31.905h9.281c3.405,0,5.028,2.916,3.993,8.361C218.433,414.1,216.665,417.159,213.3,417.868Zm.72-7.089c.581-3.057-.168-4.209-1.771-4.209h-2.238l-1.694,8.906h2.238C212.556,415.476,213.629,412.818,214.017,410.779Zm33.417,22.792.013-7.046h-6.828l-2.7,7.046h-4.939l12.694-31.905h6.6l.432,31.905Zm.09-25.755h-.066l-4.912,13.8h4.773Zm26.512,25.755H279.9L292.3,401.665h-4.939l-8.585,23.2h-.067l.7-23.2h-5.2Zm45.056,0,.932-4.9H311.7l1.768-9.3H320.9l.932-4.9h-7.424l1.5-7.892h8.055l.932-4.9h-13.06l-6.066,31.905Zm27.555.5c4.681,0,7.76-2.744,9.379-11.256h-4.949c-.545,2.511-1.2,6.476-3.52,6.476-2.821,0-2.515-3.524-.967-11.674s2.584-11.674,5.4-11.674c1.495,0,1.755,1.591.95,5.834h4.9c1.5-6.852.162-10.615-4.944-10.615-8.258,0-9.787,8.051-11.392,16.493C339.936,425.944,338.39,434.072,346.647,434.072Zm46.191-16.454c-1.581,8.326-3.128,16.454-11.383,16.454s-6.713-8.129-5.13-16.454c1.6-8.413,3.128-16.455,11.385-16.455S394.438,409.205,392.838,417.618ZM386.8,405.944c-2.821,0-3.856,3.524-5.405,11.674s-1.854,11.674.967,11.674,3.856-3.524,5.406-11.674S389.621,405.944,386.8,405.944Z" transform="translate(-2.405)" fill="#00852a"></path>
                </g>
            </svg>
        </div>
    </div>

    <div class="tabs">
        <button class="tab-btn active" onclick="openTab(event, 'charts')">Diagramme</button>
        <button class="tab-btn" onclick="openTab(event, 'table')">Datentabelle</button>
    </div>

    <!-- Charts Tab -->
    <div id="charts" class="tab-content active">
        <div class="legend-info">
            <span class="legend-actual">&#9632; Ist-Werte</span>
            <span class="legend-forecast">&#9670; Prognose</span>
            <span style="color:#6b7280">| Gestrichelte Linie = Prognosestart (Okt 2025)</span>
        </div>

        <div class="section-title">Operative Kennzahlen (aus Auftragsdaten)</div>
        <div class="grid">
"""

    # Add operational charts
    for metric in ['total_orders', 'revenue_total']:
        fig = figures[metric]
        chart_html = fig.to_html(full_html=False, include_plotlyjs=False)
        html_content += f"""
            <div class="chart-container">
                {chart_html}
            </div>
"""

    html_content += """
        </div>

        <div class="section-title">Finanzkennzahlen (aus Buchhaltung)</div>
        <div class="grid">
"""

    # Add financial charts
    for metric in ['total_revenue', 'personnel_costs', 'external_driver_costs']:
        fig = figures[metric]
        chart_html = fig.to_html(full_html=False, include_plotlyjs=False)
        html_content += f"""
            <div class="chart-container">
                {chart_html}
            </div>
"""

    html_content += """
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
                    <td>XGBoost</td>
                    <td>~5%</td>
                </tr>
                <tr>
                    <td>Betriebsertrag (SK 35060000+35070000)</td>
                    <td>Finanzen</td>
                    <td>XGBoost</td>
                    <td>2.59%</td>
                </tr>
                <tr>
                    <td>Personalaufwand (SK 0151)</td>
                    <td>Finanzen + Personal</td>
                    <td>XGBoost+Personal</td>
                    <td>2.55%</td>
                </tr>
                <tr>
                    <td>Ausgangsfrachten LKW (SK 62802000)</td>
                    <td>Finanzen</td>
                    <td>XGBoost</td>
                    <td>5.73%</td>
                </tr>
            </table>
        </div>
    </div>

    <!-- Table Tab -->
    <div id="table" class="tab-content">
        <div class="legend-info">
            <span class="legend-actual">&#9632; Ist-Werte (Jan 2022 - Sep 2025)</span>
            <span class="legend-forecast">&#9670; Prognose (Okt 2025 - Dez 2026)</span>
        </div>

"""

    # Add yearly summary table
    html_content += yearly_summary_html

    html_content += """
        <div style="margin-top:30px;padding-left:10px;border-left:4px solid #3b82f6;margin-bottom:15px;font-size:1.1rem;font-weight:600;color:#374151;">Monatliche Details</div>
        <div class="table-container">
"""

    html_content += table_html

    html_content += """
        </div>
    </div>

    <div class="footer">
        <p><strong>Traveco Transporte AG</strong> | Prognose Dashboard</p>
        <p style="margin-top:5px;opacity:0.9;font-size:0.8rem">Hinweis: Betriebsertrag wird als Absolutwert angezeigt (Original-Buchhaltungswerte sind negativ)</p>
    </div>

    <script>
        function openTab(evt, tabName) {
            var i, tabcontent, tablinks;
            tabcontent = document.getElementsByClassName("tab-content");
            for (i = 0; i < tabcontent.length; i++) {
                tabcontent[i].className = tabcontent[i].className.replace(" active", "");
            }
            tablinks = document.getElementsByClassName("tab-btn");
            for (i = 0; i < tablinks.length; i++) {
                tablinks[i].className = tablinks[i].className.replace(" active", "");
            }
            document.getElementById(tabName).className += " active";
            evt.currentTarget.className += " active";

            // Trigger resize for Plotly charts when switching to charts tab
            if (tabName === 'charts') {
                window.dispatchEvent(new Event('resize'));
            }
        }
    </script>
</body>
</html>
"""

    return html_content


def main():
    """Main pipeline."""
    print("=" * 70)
    print("Erstelle Traveco Prognose Dashboard (5 Kennzahlen)")
    print("=" * 70)

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    print("\n1. Erstelle Dashboard...")
    html_content = create_dashboard()

    output_file = OUTPUT_PATH / "forecast_dashboard_2025_2026.html"
    print(f"\n2. Speichere Dashboard: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print("\n" + "=" * 70)
    print("Dashboard erfolgreich erstellt!")
    print(f"Im Browser öffnen: {output_file}")
    print("=" * 70)

    return output_file


if __name__ == "__main__":
    main()
