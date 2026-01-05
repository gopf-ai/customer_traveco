"""Dashboard generation module

Creates interactive HTML dashboard from the single source of truth.
Reads forecasts_latest.csv which contains both operational and financial forecasts.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
from src.utils.logging_config import get_logger
from src.utils.config import ConfigLoader

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


logger = get_logger(__name__)


# Traveco brand colors
COLORS = {
    'primary': '#006633',      # Traveco green
    'secondary': '#004d26',    # Dark green
    'accent': '#00994d',       # Light green
    'actual': '#006633',       # Green for actuals
    'forecast': '#ff6600',     # Orange for forecasts
    'gray': '#666666',
    'light_gray': '#cccccc'
}


class DashboardGenerator:
    """
    Generate interactive HTML dashboard from forecasts

    Reads from:
    - data/output/forecasts_latest.csv (single source of truth)

    Generates:
    - results/dashboard.html (interactive Plotly dashboard)
    """

    # Metric display configuration
    METRIC_CONFIG = {
        # Operational metrics
        'total_orders': {
            'name': 'Aufträge (Orders)',
            'format': '{:,.0f}',
            'pipeline': 'operational'
        },
        'revenue_total': {
            'name': 'Umsatz (Revenue)',
            'format': 'CHF {:,.0f}',
            'pipeline': 'operational'
        },
        'total_drivers': {
            'name': 'Fahrer Total',
            'format': '{:,.0f}',
            'pipeline': 'operational'
        },
        'external_drivers': {
            'name': 'Externe Fahrer',
            'format': '{:,.0f}',
            'pipeline': 'operational'
        },
        'total_km_billed': {
            'name': 'KM (Fakturiert)',
            'format': '{:,.0f} km',
            'pipeline': 'operational'
        },
        # Financial metrics
        'total_betriebsertrag': {
            'name': 'Betriebsertrag',
            'format': 'CHF {:,.0f}',
            'pipeline': 'financial'
        },
        'total_revenue': {
            'name': 'Transport + Logistik',
            'format': 'CHF {:,.0f}',
            'pipeline': 'financial'
        },
        'personnel_costs': {
            'name': 'Personalaufwand',
            'format': 'CHF {:,.0f}',
            'pipeline': 'financial'
        },
        'external_driver_costs': {
            'name': 'Ausgangsfrachten LKW',
            'format': 'CHF {:,.0f}',
            'pipeline': 'financial'
        },
        'ebt': {
            'name': 'EBT',
            'format': 'CHF {:,.0f}',
            'pipeline': 'financial'
        }
    }

    def __init__(self, config: Optional[ConfigLoader] = None):
        """Initialize dashboard generator"""
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for dashboard. Install with: pip install plotly")

        self.config = config if config else ConfigLoader()
        self.output_path = Path(self.config.get('data.output_path', 'data/output'))
        self.results_path = Path(self.config.get('results.path', 'results'))

        if not self.output_path.is_absolute():
            project_root = Path(__file__).parent.parent
            self.output_path = project_root / self.output_path

        if not self.results_path.is_absolute():
            project_root = Path(__file__).parent.parent
            self.results_path = project_root / self.results_path

        self.results_path.mkdir(parents=True, exist_ok=True)

    def run(self) -> Path:
        """
        Generate the dashboard

        Returns:
            Path to generated HTML file
        """
        logger.info("=" * 60)
        logger.info("DASHBOARD GENERATION")
        logger.info("=" * 60)

        # Step 1: Load forecast data
        df = self._load_forecasts()

        # Step 2: Generate charts
        fig = self._create_dashboard(df)

        # Step 3: Save to HTML
        output_file = self.results_path / 'dashboard.html'
        fig.write_html(
            str(output_file),
            include_plotlyjs=True,
            full_html=True
        )

        logger.info(f"Dashboard saved to: {output_file}")

        return output_file

    def _load_forecasts(self) -> pd.DataFrame:
        """Load the single source of truth"""
        input_file = self.output_path / 'forecasts_latest.csv'

        if not input_file.exists():
            raise FileNotFoundError(
                f"Forecasts file not found: {input_file}\n"
                f"Run the pipeline first to generate forecasts."
            )

        df = pd.read_csv(input_file)
        df['date'] = pd.to_datetime(df['date'])

        logger.info(f"Loaded {len(df)} forecast records")
        logger.info(f"  Metrics: {df['metric'].nunique()}")
        logger.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")

        return df

    def _create_dashboard(self, df: pd.DataFrame) -> go.Figure:
        """Create the complete dashboard figure"""
        metrics = df['metric'].unique()
        n_metrics = len(metrics)

        # Calculate grid size
        n_cols = 2
        n_rows = (n_metrics + 1) // 2

        # Create subplot titles
        subplot_titles = []
        for metric in metrics:
            config = self.METRIC_CONFIG.get(metric, {'name': metric})
            subplot_titles.append(config['name'])

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=subplot_titles,
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )

        # Add traces for each metric
        for idx, metric in enumerate(metrics):
            row = (idx // n_cols) + 1
            col = (idx % n_cols) + 1

            self._add_metric_chart(fig, df, metric, row, col)

        # Update layout
        fig.update_layout(
            title={
                'text': f'<b>Traveco Forecast Dashboard</b><br><sup>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</sup>',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': COLORS['primary']}
            },
            height=400 * n_rows,
            showlegend=True,
            legend={
                'orientation': 'h',
                'yanchor': 'bottom',
                'y': 1.02,
                'xanchor': 'center',
                'x': 0.5
            },
            template='plotly_white',
            font={'family': 'Arial, sans-serif'}
        )

        return fig

    def _add_metric_chart(
        self,
        fig: go.Figure,
        df: pd.DataFrame,
        metric: str,
        row: int,
        col: int
    ):
        """Add chart for a single metric"""
        metric_df = df[df['metric'] == metric].copy()
        metric_df = metric_df.sort_values('date')

        config = self.METRIC_CONFIG.get(metric, {'name': metric, 'format': '{:,.0f}'})

        # Split into actuals and forecasts
        actuals = metric_df[metric_df['source'] == 'actual']
        forecasts = metric_df[metric_df['source'] == 'forecast']

        # Get model name and MAPE for forecast
        model_name = forecasts['model'].iloc[0] if len(forecasts) > 0 else 'N/A'
        mape = forecasts['mape'].iloc[0] if len(forecasts) > 0 else np.nan

        # Add actuals trace
        fig.add_trace(
            go.Scatter(
                x=actuals['date'],
                y=actuals['value'],
                mode='lines+markers',
                name='Ist-Werte (Actuals)',
                line={'color': COLORS['actual'], 'width': 2},
                marker={'size': 6},
                showlegend=(row == 1 and col == 1),
                legendgroup='actuals',
                hovertemplate=(
                    '<b>%{x|%b %Y}</b><br>'
                    f'{config["name"]}: ' + config["format"].replace('{', '%{y:').replace('}', '}') + '<br>'
                    '<extra></extra>'
                )
            ),
            row=row,
            col=col
        )

        # Add forecasts trace
        if len(forecasts) > 0:
            # Connect forecast to last actual
            if len(actuals) > 0:
                last_actual = actuals.iloc[-1]
                forecast_with_connection = pd.concat([
                    pd.DataFrame([{
                        'date': last_actual['date'],
                        'value': last_actual['value']
                    }]),
                    forecasts[['date', 'value']]
                ]).reset_index(drop=True)
            else:
                forecast_with_connection = forecasts

            # Annotation for model
            annotation_text = f"{model_name}"
            if not np.isnan(mape):
                annotation_text += f" (MAPE: {mape:.1f}%)"

            fig.add_trace(
                go.Scatter(
                    x=forecast_with_connection['date'],
                    y=forecast_with_connection['value'],
                    mode='lines+markers',
                    name=f'Prognose ({model_name})',
                    line={'color': COLORS['forecast'], 'width': 2, 'dash': 'dash'},
                    marker={'size': 6},
                    showlegend=(row == 1 and col == 1),
                    legendgroup='forecast',
                    hovertemplate=(
                        '<b>%{x|%b %Y}</b><br>'
                        f'{config["name"]}: ' + config["format"].replace('{', '%{y:').replace('}', '}') + '<br>'
                        f'Model: {model_name}<br>'
                        '<extra></extra>'
                    )
                ),
                row=row,
                col=col
            )

            # Add annotation for model
            fig.add_annotation(
                x=forecasts['date'].iloc[len(forecasts)//2],
                y=forecasts['value'].max(),
                text=annotation_text,
                showarrow=False,
                font={'size': 10, 'color': COLORS['forecast']},
                row=row,
                col=col
            )

        # Format y-axis
        fig.update_yaxes(
            title_text='',
            tickformat=',.0f',
            row=row,
            col=col
        )

        # Format x-axis
        fig.update_xaxes(
            title_text='',
            tickformat='%b %Y',
            row=row,
            col=col
        )


def combine_pipeline_outputs(config: Optional[ConfigLoader] = None) -> pd.DataFrame:
    """
    Combine operational and financial forecasts into single file

    This is the final step that creates the single source of truth.
    """
    if config is None:
        config = ConfigLoader()

    output_path = Path(config.get('data.output_path', 'data/output'))

    if not output_path.is_absolute():
        project_root = Path(__file__).parent.parent
        output_path = project_root / output_path

    logger.info("Combining pipeline outputs...")

    all_dfs = []

    # Load operational forecasts
    op_file = output_path / 'operational_forecasts.csv'
    if op_file.exists():
        df_op = pd.read_csv(op_file)
        all_dfs.append(df_op)
        logger.info(f"  Loaded {len(df_op)} operational records")

    # Load financial forecasts
    fin_file = output_path / 'financial_forecasts.csv'
    if fin_file.exists():
        df_fin = pd.read_csv(fin_file)
        all_dfs.append(df_fin)
        logger.info(f"  Loaded {len(df_fin)} financial records")

    if not all_dfs:
        raise FileNotFoundError("No forecast files found. Run the pipelines first.")

    # Combine
    df_combined = pd.concat(all_dfs, ignore_index=True)
    df_combined['date'] = pd.to_datetime(df_combined['date'])
    df_combined = df_combined.sort_values(['pipeline', 'metric', 'date']).reset_index(drop=True)

    # Save
    output_file = output_path / 'forecasts_latest.csv'
    df_combined.to_csv(output_file, index=False)
    logger.info(f"Saved combined forecasts to: {output_file}")
    logger.info(f"  Total records: {len(df_combined)}")

    return df_combined


def run():
    """Entry point for dashboard generation"""
    # First combine outputs
    combine_pipeline_outputs()

    # Then generate dashboard
    generator = DashboardGenerator()
    return generator.run()


if __name__ == '__main__':
    run()
