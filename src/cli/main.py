"""Main CLI interface for Traveco forecasting system

Provides command-line commands for:
- Training forecasting models
- Generating forecasts
- Validating forecasts
- Generating business reports
- Checking system status
"""

import click
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint
import pandas as pd
from datetime import datetime
import json

from src.pipeline import ForecastingPipeline
from src.utils.logging_config import get_logger


console = Console()
logger = get_logger(__name__)


@click.group()
@click.version_option(version='1.0.0', prog_name='Traveco Forecasting System')
def cli():
    """
    Traveco Forecasting System

    ML-based forecasting for transport logistics metrics:
    - Revenue (transportation + total)
    - Personnel costs
    - External drivers
    """
    pass


@cli.command()
@click.option(
    '--config',
    '-c',
    type=click.Path(exists=True),
    default='config/config.yaml',
    help='Path to configuration file'
)
@click.option(
    '--start-date',
    type=str,
    help='Start date for training data (YYYY-MM-DD)'
)
@click.option(
    '--end-date',
    type=str,
    help='End date for training data (YYYY-MM-DD)'
)
@click.option(
    '--skip-baseline/--no-skip-baseline',
    default=False,
    help='Skip baseline models (Seasonal Naive, MA, Linear Trend)'
)
@click.option(
    '--skip-xgboost/--no-skip-xgboost',
    default=False,
    help='Skip XGBoost models'
)
@click.option(
    '--skip-revenue/--no-skip-revenue',
    default=False,
    help='Skip revenue percentage models'
)
@click.option(
    '--no-save',
    is_flag=True,
    help='Do not save trained models to disk'
)
def train(config, start_date, end_date, skip_baseline, skip_xgboost, skip_revenue, no_save):
    """
    Train forecasting models on historical data

    Example:
        traveco-forecast train
        traveco-forecast train --start-date 2022-01-01 --end-date 2024-12-31
        traveco-forecast train --skip-baseline
    """
    console.print(Panel.fit(
        "[bold cyan]Traveco Forecasting System - Model Training[/bold cyan]",
        border_style="cyan"
    ))

    try:
        # Initialize pipeline
        console.print("\n[yellow]Initializing pipeline...[/yellow]")
        pipeline = ForecastingPipeline(config_path=config)

        # Load data
        console.print("\n[yellow]Loading historical data...[/yellow]")
        df_train = pipeline.load_data(
            start_date=start_date,
            end_date=end_date,
            validate=True
        )

        console.print(f"[green]✓[/green] Loaded {len(df_train)} months of data")
        console.print(f"  Date range: {df_train['date'].min()} to {df_train['date'].max()}")

        # Train models
        console.print("\n[yellow]Training models...[/yellow]")
        models = pipeline.train_models(
            df_train=df_train,
            train_baseline=not skip_baseline,
            train_xgboost=not skip_xgboost,
            train_revenue=not skip_revenue,
            save_models=not no_save
        )

        # Display results
        console.print("\n[bold green]✓ Training Complete![/bold green]\n")

        table = Table(title="Trained Models")
        table.add_column("Metric", style="cyan")
        table.add_column("Models", style="green")

        for metric, metric_models in models.items():
            model_list = ", ".join(metric_models.keys())
            table.add_row(metric, model_list)

        console.print(table)

        if pipeline.revenue_ensemble:
            weights = pipeline.revenue_ensemble.get_weights()
            console.print(f"\n[bold]Revenue Ensemble Weights:[/bold]")
            console.print(f"  Simple Model: {weights['percentage_model']:.1%}")
            console.print(f"  ML Model:     {weights['ml_model']:.1%}")

    except Exception as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {str(e)}")
        logger.exception("Training failed")
        raise click.Abort()


@cli.command()
@click.option(
    '--config',
    '-c',
    type=click.Path(exists=True),
    default='config/config.yaml',
    help='Path to configuration file'
)
@click.option(
    '--year',
    '-y',
    type=int,
    required=True,
    help='Year to forecast'
)
@click.option(
    '--months',
    '-m',
    type=int,
    default=12,
    help='Number of months to forecast'
)
@click.option(
    '--model',
    type=click.Choice(['xgboost', 'seasonal_naive', 'moving_average', 'linear_trend']),
    default='xgboost',
    help='Model to use for forecasting'
)
@click.option(
    '--skip-revenue',
    is_flag=True,
    help='Skip total revenue forecast'
)
@click.option(
    '--output',
    '-o',
    type=click.Path(),
    help='Output file path (auto-generated if not specified)'
)
@click.option(
    '--format',
    type=click.Choice(['csv', 'excel', 'json']),
    default='csv',
    help='Output format'
)
def forecast(config, year, months, model, skip_revenue, output, format):
    """
    Generate forecasts for specified period

    Example:
        traveco-forecast forecast --year 2025 --months 12
        traveco-forecast forecast -y 2026 -m 6 --model seasonal_naive
        traveco-forecast forecast -y 2025 --output forecasts/2025_budget.csv
    """
    console.print(Panel.fit(
        f"[bold cyan]Generating {months}-Month Forecast for {year}[/bold cyan]",
        border_style="cyan"
    ))

    try:
        # Initialize pipeline
        console.print("\n[yellow]Initializing pipeline...[/yellow]")
        pipeline = ForecastingPipeline(config_path=config)

        # Load historical data
        console.print("[yellow]Loading historical data...[/yellow]")
        pipeline.load_data()

        # Load models
        console.print("[yellow]Loading trained models...[/yellow]")
        pipeline.load_models()

        # Generate forecast
        console.print(f"\n[yellow]Generating forecast using {model} model...[/yellow]")
        df_forecast = pipeline.generate_forecast(
            year=year,
            n_months=months,
            model_type=model,
            include_revenue_forecast=not skip_revenue,
            save_forecast=False  # We'll save manually with custom path
        )

        # Save forecast
        if output is None:
            output = f"forecasts/forecast_{year}_{model}_{datetime.now().strftime('%Y%m%d')}.{format}"

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'csv':
            df_forecast.to_csv(output_path, index=False)
        elif format == 'excel':
            df_forecast.to_excel(output_path, index=False)
        elif format == 'json':
            df_forecast.to_json(output_path, orient='records', date_format='iso')

        console.print(f"\n[bold green]✓ Forecast saved to:[/bold green] {output_path}")

        # Display summary
        table = Table(title="Forecast Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Total", style="green", justify="right")
        table.add_column("Monthly Avg", style="yellow", justify="right")

        for col in df_forecast.columns:
            if col in ['date', 'model_type', 'forecast_date']:
                continue

            if pd.api.types.is_numeric_dtype(df_forecast[col]):
                total = df_forecast[col].sum()
                avg = df_forecast[col].mean()

                # Format based on magnitude
                if total > 1000000:
                    total_str = f"CHF {total:,.0f}"
                    avg_str = f"CHF {avg:,.0f}"
                else:
                    total_str = f"{total:,.0f}"
                    avg_str = f"{avg:,.0f}"

                table.add_row(col, total_str, avg_str)

        console.print("\n")
        console.print(table)

    except Exception as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {str(e)}")
        logger.exception("Forecast generation failed")
        raise click.Abort()


@cli.command()
@click.option(
    '--config',
    '-c',
    type=click.Path(exists=True),
    default='config/config.yaml',
    help='Path to configuration file'
)
@click.option(
    '--forecast',
    '-f',
    type=click.Path(exists=True),
    required=True,
    help='Path to forecast file'
)
@click.option(
    '--actual',
    '-a',
    type=click.Path(exists=True),
    required=True,
    help='Path to actual data file'
)
@click.option(
    '--metrics',
    '-m',
    multiple=True,
    help='Metrics to validate (validates all if not specified)'
)
@click.option(
    '--output',
    '-o',
    type=click.Path(),
    help='Output file for validation report'
)
def validate(config, forecast, actual, metrics, output):
    """
    Validate forecasts against actual data

    Example:
        traveco-forecast validate -f forecasts/2025_forecast.csv -a data/actual_2025.csv
        traveco-forecast validate -f forecast.csv -a actual.csv -m revenue_total -m external_drivers
        traveco-forecast validate -f forecast.csv -a actual.csv -o reports/validation.json
    """
    console.print(Panel.fit(
        "[bold cyan]Forecast Validation[/bold cyan]",
        border_style="cyan"
    ))

    try:
        # Initialize pipeline
        pipeline = ForecastingPipeline(config_path=config)

        # Load data
        console.print("\n[yellow]Loading forecast and actual data...[/yellow]")
        df_forecast = pd.read_csv(forecast)
        df_actual = pd.read_csv(actual)

        console.print(f"[green]✓[/green] Loaded {len(df_forecast)} forecast periods")
        console.print(f"[green]✓[/green] Loaded {len(df_actual)} actual periods")

        # Validate
        console.print("\n[yellow]Calculating validation metrics...[/yellow]")
        metrics_list = list(metrics) if metrics else None
        validation_results = pipeline.validate_forecast(
            df_forecast=df_forecast,
            df_actual=df_actual,
            metrics_to_validate=metrics_list
        )

        # Display results
        console.print("\n[bold green]✓ Validation Complete![/bold green]\n")

        table = Table(title="Validation Results")
        table.add_column("Metric", style="cyan")
        table.add_column("MAPE", style="yellow", justify="right")
        table.add_column("MAE", style="green", justify="right")
        table.add_column("RMSE", style="blue", justify="right")
        table.add_column("R²", style="magenta", justify="right")

        for metric, results in validation_results.items():
            table.add_row(
                metric,
                f"{results['mape']:.2f}%",
                f"{results['mae']:,.0f}",
                f"{results['rmse']:,.0f}",
                f"{results['r2']:.3f}"
            )

        console.print(table)

        # Save results if output specified
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(validation_results, f, indent=2)

            console.print(f"\n[green]✓[/green] Validation report saved to: {output_path}")

    except Exception as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {str(e)}")
        logger.exception("Validation failed")
        raise click.Abort()


@cli.command()
@click.option(
    '--config',
    '-c',
    type=click.Path(exists=True),
    default='config/config.yaml',
    help='Path to configuration file'
)
@click.option(
    '--forecast',
    '-f',
    type=click.Path(exists=True),
    required=True,
    help='Path to forecast file'
)
@click.option(
    '--output',
    '-o',
    type=click.Path(),
    required=True,
    help='Output file path'
)
@click.option(
    '--format',
    type=click.Choice(['json', 'csv', 'excel']),
    default='json',
    help='Report format'
)
def report(config, forecast, output, format):
    """
    Generate business-friendly forecast report

    Example:
        traveco-forecast report -f forecasts/2025_forecast.csv -o reports/2025_summary.json
        traveco-forecast report -f forecast.csv -o report.xlsx --format excel
    """
    console.print(Panel.fit(
        "[bold cyan]Generating Business Report[/bold cyan]",
        border_style="cyan"
    ))

    try:
        # Initialize pipeline
        pipeline = ForecastingPipeline(config_path=config)

        # Load forecast
        console.print("\n[yellow]Loading forecast data...[/yellow]")
        df_forecast = pd.read_csv(forecast)
        console.print(f"[green]✓[/green] Loaded {len(df_forecast)} forecast periods")

        # Load models if available (for revenue ensemble weights)
        try:
            pipeline.load_models()
        except:
            pass

        # Generate report
        console.print("\n[yellow]Generating report...[/yellow]")
        report_data = pipeline.generate_business_report(
            df_forecast=df_forecast,
            output_path=output,
            format=format
        )

        console.print(f"\n[bold green]✓ Report saved to:[/bold green] {output}")

        # Display summary
        console.print("\n[bold]Forecast Summary:[/bold]")
        console.print(f"  Period: {report_data['forecast_metadata']['start_date']} to {report_data['forecast_metadata']['end_date']}")
        console.print(f"  Months: {report_data['forecast_metadata']['n_periods']}")
        console.print(f"  Model:  {report_data['forecast_metadata']['model_type']}")

        if 'metrics' in report_data:
            table = Table(title="Metrics Overview")
            table.add_column("Metric", style="cyan")
            table.add_column("Total", style="green", justify="right")
            table.add_column("Monthly Avg", style="yellow", justify="right")

            for metric, values in report_data['metrics'].items():
                total = values['total']
                avg = values['monthly_average']

                if total > 1000000:
                    total_str = f"CHF {total:,.0f}"
                    avg_str = f"CHF {avg:,.0f}"
                else:
                    total_str = f"{total:,.0f}"
                    avg_str = f"{avg:,.0f}"

                table.add_row(metric, total_str, avg_str)

            console.print("\n")
            console.print(table)

    except Exception as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {str(e)}")
        logger.exception("Report generation failed")
        raise click.Abort()


@cli.command()
@click.option(
    '--config',
    '-c',
    type=click.Path(exists=True),
    default='config/config.yaml',
    help='Path to configuration file'
)
def status(config):
    """
    Show system status and model information

    Example:
        traveco-forecast status
    """
    console.print(Panel.fit(
        "[bold cyan]Traveco Forecasting System - Status[/bold cyan]",
        border_style="cyan"
    ))

    try:
        # Initialize pipeline
        pipeline = ForecastingPipeline(config_path=config)

        # Check for trained models
        console.print("\n[bold]Checking for trained models...[/bold]")

        models_dir = Path(pipeline.config.get('paths.models_dir', 'models'))

        if not models_dir.exists():
            console.print("[yellow]⚠[/yellow] No models directory found")
            console.print("  Run 'traveco-forecast train' to train models")
            return

        model_files = list(models_dir.glob('*.pkl'))

        if not model_files:
            console.print("[yellow]⚠[/yellow] No trained models found")
            console.print("  Run 'traveco-forecast train' to train models")
            return

        # Display found models
        table = Table(title="Available Models")
        table.add_column("Model File", style="cyan")
        table.add_column("Size", style="green", justify="right")
        table.add_column("Modified", style="yellow")

        for model_file in sorted(model_files):
            size = model_file.stat().st_size
            size_str = f"{size / 1024:.1f} KB" if size < 1024 * 1024 else f"{size / (1024 * 1024):.1f} MB"

            modified = datetime.fromtimestamp(model_file.stat().st_mtime)
            modified_str = modified.strftime('%Y-%m-%d %H:%M')

            table.add_row(model_file.name, size_str, modified_str)

        console.print("\n")
        console.print(table)

        # Check data availability
        console.print("\n[bold]Checking data sources...[/bold]")

        data_dir = Path(pipeline.config.get('paths.raw_data_dir', 'data/raw'))

        required_files = [
            pipeline.config.get('data_files.orders_file'),
            pipeline.config.get('data_files.tours_file'),
            pipeline.config.get('data_files.working_days_file')
        ]

        optional_files = [
            pipeline.config.get('data_files.personnel_costs_file'),
            pipeline.config.get('data_files.total_revenue_file')
        ]

        for file in required_files:
            if file:
                file_path = data_dir / file
                if file_path.exists():
                    console.print(f"  [green]✓[/green] {file}")
                else:
                    console.print(f"  [red]✗[/red] {file} (required)")

        for file in optional_files:
            if file:
                file_path = data_dir / file
                if file_path.exists():
                    console.print(f"  [green]✓[/green] {file}")
                else:
                    console.print(f"  [yellow]⚠[/yellow] {file} (optional)")

        # Configuration info
        console.print("\n[bold]Configuration:[/bold]")
        console.print(f"  Config file: {config}")
        console.print(f"  Models dir:  {models_dir}")
        console.print(f"  Data dir:    {data_dir}")

        core_metrics = pipeline.config.get('features.core_metrics', [])
        console.print(f"  Core metrics: {', '.join(core_metrics)}")

    except Exception as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {str(e)}")
        logger.exception("Status check failed")
        raise click.Abort()


if __name__ == '__main__':
    cli()
