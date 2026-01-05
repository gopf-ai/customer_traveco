#!/usr/bin/env python3
"""
Traveco Forecasting Pipeline
============================

Single entry point to run the complete forecasting pipeline.

Usage:
    python run_pipeline.py                    # Run complete pipeline
    python run_pipeline.py --operational      # Run operational only
    python run_pipeline.py --financial        # Run financial only
    python run_pipeline.py --dashboard        # Generate dashboard only

Pipeline Flow:
    1. Operational Data Prep    -> data/intermediate/operational_time_series.csv
    2. Operational Model Eval   -> data/intermediate/operational_model_eval.csv
    3. Operational Forecast     -> data/output/operational_forecasts.csv
    4. Financial Data Prep      -> data/intermediate/financial_time_series.csv
    5. Financial Model Eval     -> data/intermediate/financial_model_eval.csv
    6. Financial Forecast       -> data/output/financial_forecasts.csv
    7. Combine Outputs          -> data/output/forecasts_latest.csv (SINGLE SOURCE OF TRUTH)
    8. Generate Dashboard       -> results/dashboard.html
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def run_operational_pipeline():
    """Run the complete operational pipeline"""
    print("\n" + "=" * 70)
    print("OPERATIONAL PIPELINE")
    print("=" * 70)

    from src.operational import run_data_prep, run_model_eval, run_forecast

    print("\n[1/3] Data Preparation...")
    run_data_prep()

    print("\n[2/3] Model Evaluation...")
    run_model_eval()

    print("\n[3/3] Forecast Generation...")
    run_forecast()

    print("\nOperational pipeline complete.")


def run_financial_pipeline():
    """Run the complete financial pipeline"""
    print("\n" + "=" * 70)
    print("FINANCIAL PIPELINE")
    print("=" * 70)

    from src.financial import run_data_prep, run_model_eval, run_forecast

    print("\n[1/3] Data Preparation...")
    run_data_prep()

    print("\n[2/3] Model Evaluation...")
    run_model_eval()

    print("\n[3/3] Forecast Generation...")
    run_forecast()

    print("\nFinancial pipeline complete.")


def run_dashboard():
    """Generate the dashboard from combined outputs"""
    print("\n" + "=" * 70)
    print("DASHBOARD GENERATION")
    print("=" * 70)

    from src.dashboard import combine_pipeline_outputs, DashboardGenerator

    print("\n[1/2] Combining pipeline outputs...")
    combine_pipeline_outputs()

    print("\n[2/2] Generating dashboard...")
    generator = DashboardGenerator()
    output_path = generator.run()

    print(f"\nDashboard saved to: {output_path}")


def run_full_pipeline():
    """Run the complete pipeline: operational + financial + dashboard"""
    start_time = datetime.now()

    print("\n" + "=" * 70)
    print("TRAVECO FORECASTING PIPELINE")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Run operational pipeline
    try:
        run_operational_pipeline()
    except Exception as e:
        print(f"\nERROR in operational pipeline: {e}")
        print("Continuing with financial pipeline...")

    # Run financial pipeline
    try:
        run_financial_pipeline()
    except Exception as e:
        print(f"\nERROR in financial pipeline: {e}")
        print("Continuing with dashboard generation...")

    # Generate dashboard
    try:
        run_dashboard()
    except Exception as e:
        print(f"\nERROR in dashboard generation: {e}")

    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print(f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration:.1f} seconds")
    print("=" * 70)

    print("\nOutputs:")
    print("  - data/intermediate/operational_time_series.csv")
    print("  - data/intermediate/operational_model_eval.csv")
    print("  - data/intermediate/financial_time_series.csv")
    print("  - data/intermediate/financial_model_eval.csv")
    print("  - data/output/operational_forecasts.csv")
    print("  - data/output/financial_forecasts.csv")
    print("  - data/output/forecasts_latest.csv  (SINGLE SOURCE OF TRUTH)")
    print("  - results/dashboard.html")


def main():
    parser = argparse.ArgumentParser(
        description="Traveco Forecasting Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_pipeline.py                  # Run complete pipeline
    python run_pipeline.py --operational    # Run operational only
    python run_pipeline.py --financial      # Run financial only
    python run_pipeline.py --dashboard      # Generate dashboard only
        """
    )

    parser.add_argument(
        '--operational', '-o',
        action='store_true',
        help='Run operational pipeline only'
    )

    parser.add_argument(
        '--financial', '-f',
        action='store_true',
        help='Run financial pipeline only'
    )

    parser.add_argument(
        '--dashboard', '-d',
        action='store_true',
        help='Generate dashboard only (requires existing forecast outputs)'
    )

    args = parser.parse_args()

    # Determine what to run
    if args.dashboard:
        run_dashboard()
    elif args.operational:
        run_operational_pipeline()
    elif args.financial:
        run_financial_pipeline()
    else:
        # Run full pipeline
        run_full_pipeline()


if __name__ == '__main__':
    main()
