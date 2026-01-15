#!/usr/bin/env python3
"""
Time Series Demo Runner
=======================

Run time series analysis demos from the command line.

Usage:
    python -m ts_examples.run --demo arima
    python -m ts_examples.run --demo ets
    python -m ts_examples.run --list
    python -m ts_examples.run --all

Available demos:
    arima       - ARIMA model fitting, diagnostics, and forecasting
    ets         - Exponential smoothing (SES, Holt, Holt-Winters)
    stl         - STL decomposition (robust vs non-robust, comparison)
    kalman      - Kalman filter for local level/trend models
    var         - Vector Autoregression and Granger causality
    changepoint - Change-point detection (CUSUM, Binary Segmentation, PELT)
    backtest    - Rolling-origin backtesting and model comparison
    metrics     - Forecast accuracy metrics (MAE, RMSE, MAPE, MASE, etc.)
"""

import argparse
import sys
import os
from pathlib import Path


def ensure_claude_dir():
    """Ensure .claude directory exists for plot outputs."""
    claude_dir = Path('.claude')
    if not claude_dir.exists():
        claude_dir.mkdir(parents=True)
        print(f"Created {claude_dir} for temporary outputs")


def get_available_demos():
    """Get dictionary of available demos."""
    from ts_examples.demos import DEMOS
    return DEMOS


def list_demos():
    """Print list of available demos."""
    demos = get_available_demos()
    print("\nAvailable demos:")
    print("-" * 50)

    descriptions = {
        'arima': 'ARIMA model fitting, diagnostics, and forecasting',
        'ets': 'Exponential smoothing (SES, Holt, Holt-Winters)',
        'stl': 'STL decomposition (robust vs non-robust)',
        'kalman': 'Kalman filter for local level/trend models',
        'var': 'Vector Autoregression and Granger causality',
        'changepoint': 'Change-point detection (CUSUM, BinSeg, PELT)',
        'backtest': 'Rolling-origin backtesting and model comparison',
        'metrics': 'Forecast accuracy metrics (MAE, RMSE, MAPE, MASE)',
    }

    for name in sorted(demos.keys()):
        desc = descriptions.get(name, 'No description available')
        print(f"  {name:<12} - {desc}")

    print("\nUsage:")
    print("  python -m ts_examples.run --demo <name>")
    print("  python -m ts_examples.run --all")


def run_demo(name):
    """Run a specific demo by name."""
    demos = get_available_demos()

    if name not in demos:
        print(f"Error: Unknown demo '{name}'")
        print(f"Available demos: {', '.join(sorted(demos.keys()))}")
        return False

    ensure_claude_dir()

    print(f"\n{'#'*60}")
    print(f"# Running demo: {name}")
    print(f"{'#'*60}\n")

    try:
        demo_module = demos[name]
        demo_module.run_demo()
        return True
    except Exception as e:
        print(f"\nError running demo '{name}': {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_demos():
    """Run all available demos."""
    demos = get_available_demos()
    ensure_claude_dir()

    results = {}
    for name in sorted(demos.keys()):
        print(f"\n{'='*60}")
        print(f"Running demo: {name}")
        print(f"{'='*60}")

        try:
            demos[name].run_demo()
            results[name] = 'SUCCESS'
        except Exception as e:
            print(f"Error: {e}")
            results[name] = f'FAILED: {e}'

    # Summary
    print("\n" + "="*60)
    print("DEMO SUMMARY")
    print("="*60)
    for name, status in results.items():
        symbol = '✓' if status == 'SUCCESS' else '✗'
        print(f"  {symbol} {name}: {status}")

    success_count = sum(1 for s in results.values() if s == 'SUCCESS')
    print(f"\nCompleted: {success_count}/{len(results)} demos successful")

    return all(s == 'SUCCESS' for s in results.values())


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run time series analysis demos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m ts_examples.run --demo arima    # Run ARIMA demo
  python -m ts_examples.run --demo ets      # Run exponential smoothing demo
  python -m ts_examples.run --list          # List all available demos
  python -m ts_examples.run --all           # Run all demos

Related documentation:
  docs/en/  - English time series notes
  docs/zh/  - Chinese time series notes
        """
    )

    parser.add_argument(
        '--demo', '-d',
        type=str,
        help='Name of demo to run (e.g., arima, ets, stl)'
    )

    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all available demos'
    )

    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Run all demos'
    )

    args = parser.parse_args()

    # Handle arguments
    if args.list:
        list_demos()
        return 0

    if args.all:
        success = run_all_demos()
        return 0 if success else 1

    if args.demo:
        success = run_demo(args.demo)
        return 0 if success else 1

    # No arguments: show help
    parser.print_help()
    print("\nTip: Use --list to see available demos")
    return 0


if __name__ == '__main__':
    sys.exit(main())
