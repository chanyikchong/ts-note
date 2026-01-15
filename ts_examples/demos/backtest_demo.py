"""
Rolling Backtest Demo
=====================
Demonstrates time series cross-validation and rolling-origin evaluation.

Related notes:
- docs/en/model-selection/cross-validation.md
- docs/en/forecasting/multi-step.md
"""

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA


def generate_data(n=200, seed=42):
    """Generate data with trend and seasonality."""
    np.random.seed(seed)
    t = np.arange(n)
    trend = 50 + 0.2 * t
    seasonal = 10 * np.sin(2 * np.pi * t / 12)
    noise = np.random.randn(n) * 3
    return trend + seasonal + noise


def compute_metrics(actual, predicted):
    """Compute forecast accuracy metrics."""
    actual = np.array(actual)
    predicted = np.array(predicted)

    errors = actual - predicted
    abs_errors = np.abs(errors)

    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(errors**2))

    # MAPE (handle zeros)
    mask = actual != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs(errors[mask] / actual[mask])) * 100
    else:
        mape = np.nan

    # SMAPE
    denominator = (np.abs(actual) + np.abs(predicted)) / 2
    mask = denominator != 0
    if mask.sum() > 0:
        smape = np.mean(np.abs(errors[mask]) / denominator[mask]) * 100
    else:
        smape = np.nan

    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'SMAPE': smape}


class RollingBacktest:
    """
    Rolling-origin backtesting framework.

    Parameters:
    -----------
    data : array-like - Full time series
    initial_train_size : int - Size of initial training window
    horizon : int - Forecast horizon
    step : int - Step size between origins
    expanding : bool - If True, use expanding window; if False, use rolling window
    """

    def __init__(self, data, initial_train_size, horizon=1, step=1, expanding=True):
        self.data = np.array(data)
        self.initial_train_size = initial_train_size
        self.horizon = horizon
        self.step = step
        self.expanding = expanding

        self.results = {}

    def run(self, model_fn, model_name):
        """
        Run backtest for a given model.

        Parameters:
        -----------
        model_fn : callable - Function that takes training data and returns forecasts
                             Signature: model_fn(train_data, horizon) -> array of forecasts
        model_name : str - Name identifier for this model
        """
        n = len(self.data)
        forecasts = []
        actuals = []
        origins = []

        # Iterate through origins
        origin = self.initial_train_size
        while origin + self.horizon <= n:
            # Get training data
            if self.expanding:
                train = self.data[:origin]
            else:
                train = self.data[origin - self.initial_train_size:origin]

            # Get forecasts
            try:
                fc = model_fn(train, self.horizon)
                fc = np.atleast_1d(fc)

                # Get actual values
                actual = self.data[origin:origin + self.horizon]

                forecasts.append(fc)
                actuals.append(actual)
                origins.append(origin)
            except Exception as e:
                print(f"  Warning: Model failed at origin {origin}: {e}")

            origin += self.step

        # Compute metrics
        all_forecasts = np.concatenate(forecasts)
        all_actuals = np.concatenate(actuals)
        metrics = compute_metrics(all_actuals, all_forecasts)

        self.results[model_name] = {
            'forecasts': forecasts,
            'actuals': actuals,
            'origins': origins,
            'metrics': metrics,
            'all_forecasts': all_forecasts,
            'all_actuals': all_actuals
        }

        return metrics

    def summary(self):
        """Print summary of all models."""
        print("\nBacktest Summary")
        print("="*60)
        print(f"{'Model':<20} {'MAE':<10} {'RMSE':<10} {'MAPE':<10} {'SMAPE':<10}")
        print("-"*60)
        for name, result in self.results.items():
            m = result['metrics']
            print(f"{name:<20} {m['MAE']:<10.4f} {m['RMSE']:<10.4f} "
                  f"{m['MAPE']:<10.2f} {m['SMAPE']:<10.2f}")


def naive_forecast(train, horizon):
    """Naive forecast: last value."""
    return np.repeat(train[-1], horizon)


def seasonal_naive_forecast(train, horizon, period=12):
    """Seasonal naive: same value from last season."""
    forecasts = []
    for h in range(horizon):
        idx = len(train) - period + (h % period)
        if idx >= 0:
            forecasts.append(train[idx])
        else:
            forecasts.append(train[-1])
    return np.array(forecasts)


def ets_forecast(train, horizon):
    """Holt-Winters forecast."""
    model = ExponentialSmoothing(
        train, trend='add', seasonal='add',
        seasonal_periods=12, initialization_method='estimated'
    )
    fit = model.fit(optimized=True)
    return fit.forecast(horizon)


def arima_forecast(train, horizon):
    """ARIMA forecast."""
    model = ARIMA(train, order=(1, 1, 1))
    fit = model.fit()
    return fit.forecast(horizon)


def demo_rolling_vs_expanding():
    """Compare rolling vs expanding window."""
    print("\n" + "="*50)
    print("Rolling vs Expanding Window Comparison")
    print("="*50)

    data = generate_data(n=150)

    # Expanding window
    bt_expanding = RollingBacktest(
        data, initial_train_size=60, horizon=12, step=12, expanding=True
    )
    bt_expanding.run(naive_forecast, "Naive (Expanding)")

    # Rolling window
    bt_rolling = RollingBacktest(
        data, initial_train_size=60, horizon=12, step=12, expanding=False
    )
    bt_rolling.run(naive_forecast, "Naive (Rolling)")

    print("\nExpanding Window Results:")
    print(f"  MAE: {bt_expanding.results['Naive (Expanding)']['metrics']['MAE']:.4f}")
    print(f"  Number of origins: {len(bt_expanding.results['Naive (Expanding)']['origins'])}")

    print("\nRolling Window Results:")
    print(f"  MAE: {bt_rolling.results['Naive (Rolling)']['metrics']['MAE']:.4f}")
    print(f"  Number of origins: {len(bt_rolling.results['Naive (Rolling)']['origins'])}")


def demo_model_comparison():
    """Compare different models via backtesting."""
    print("\n" + "="*50)
    print("Model Comparison via Backtesting")
    print("="*50)

    data = generate_data(n=150)

    # Setup backtest
    bt = RollingBacktest(
        data, initial_train_size=60, horizon=12, step=6, expanding=True
    )

    # Define model functions
    models = {
        'Naive': naive_forecast,
        'Seasonal Naive': lambda train, h: seasonal_naive_forecast(train, h, period=12),
        'Holt-Winters': ets_forecast,
        'ARIMA(1,1,1)': arima_forecast
    }

    print("\nRunning backtest...")
    for name, model_fn in models.items():
        print(f"  Testing {name}...")
        bt.run(model_fn, name)

    bt.summary()

    return bt


def demo_multi_horizon():
    """Demonstrate multi-step forecast evaluation."""
    print("\n" + "="*50)
    print("Multi-Step Forecast Evaluation")
    print("="*50)

    data = generate_data(n=150)

    print("\nMAE by forecast horizon (Holt-Winters):")
    print(f"{'Horizon':<10} {'MAE':<10}")
    print("-"*20)

    for h in [1, 3, 6, 12]:
        bt = RollingBacktest(
            data, initial_train_size=60, horizon=h, step=6, expanding=True
        )
        metrics = bt.run(ets_forecast, f"H={h}")
        print(f"{h:<10} {metrics['MAE']:<10.4f}")


def demo_leakage_warning():
    """Demonstrate data leakage pitfall."""
    print("\n" + "="*50)
    print("Data Leakage Warning Demo")
    print("="*50)

    data = generate_data(n=150)

    # WRONG: Use future data to scale
    print("\nIncorrect approach (data leakage):")
    global_mean = np.mean(data)
    global_std = np.std(data)
    data_scaled_wrong = (data - global_mean) / global_std

    # CORRECT: Use only past data to scale
    print("Correct approach (no leakage):")

    def scale_correctly(train, test):
        train_mean = np.mean(train)
        train_std = np.std(train)
        scaled_train = (train - train_mean) / train_std
        scaled_test = (test - train_mean) / train_std
        return scaled_train, scaled_test, train_mean, train_std

    train_idx = 100
    train = data[:train_idx]
    test = data[train_idx:]

    scaled_train, scaled_test, used_mean, used_std = scale_correctly(train, test)

    print(f"\n  Training set mean: {np.mean(train):.2f}")
    print(f"  Global mean: {global_mean:.2f}")
    print(f"  Test set mean (raw): {np.mean(test):.2f}")
    print(f"  Test set mean (scaled with train params): {np.mean(scaled_test):.4f}")

    print("\n  Key insight: Scaling parameters must be computed")
    print("  only from training data available at forecast time.")


def plot_backtest_results(bt, data, filename='.claude/backtest_demo_plot.png'):
    """Plot backtest results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Full series with origins marked
    axes[0, 0].plot(data, 'b-', alpha=0.7, label='Data')
    if 'Holt-Winters' in bt.results:
        origins = bt.results['Holt-Winters']['origins']
        for o in origins[:5]:  # First 5 origins
            axes[0, 0].axvline(x=o, color='r', alpha=0.3, linestyle='--')
    axes[0, 0].set_title('Time Series with Forecast Origins')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Forecast vs Actual scatter
    if 'Holt-Winters' in bt.results:
        result = bt.results['Holt-Winters']
        axes[0, 1].scatter(result['all_actuals'], result['all_forecasts'],
                          alpha=0.5, s=20)
        lims = [min(result['all_actuals'].min(), result['all_forecasts'].min()),
                max(result['all_actuals'].max(), result['all_forecasts'].max())]
        axes[0, 1].plot(lims, lims, 'r--', label='Perfect forecast')
        axes[0, 1].set_xlabel('Actual')
        axes[0, 1].set_ylabel('Forecast')
        axes[0, 1].set_title('Forecast vs Actual (Holt-Winters)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    # Error distribution
    if 'Holt-Winters' in bt.results:
        result = bt.results['Holt-Winters']
        errors = result['all_actuals'] - result['all_forecasts']
        axes[1, 0].hist(errors, bins=20, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1, 0].set_title('Forecast Error Distribution')
        axes[1, 0].set_xlabel('Error (Actual - Forecast)')
        axes[1, 0].grid(True, alpha=0.3)

    # Model comparison bar chart
    models = list(bt.results.keys())
    maes = [bt.results[m]['metrics']['MAE'] for m in models]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(models)))
    axes[1, 1].bar(range(len(models)), maes, color=colors)
    axes[1, 1].set_xticks(range(len(models)))
    axes[1, 1].set_xticklabels(models, rotation=45, ha='right')
    axes[1, 1].set_ylabel('MAE')
    axes[1, 1].set_title('Model Comparison (MAE)')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def run_demo():
    """Run the complete backtest demo."""
    print("="*60)
    print("Rolling Backtest Demonstration")
    print("="*60)

    # Generate data
    data = generate_data()
    print(f"\nGenerated {len(data)} observations with trend and seasonality")

    # Run demos
    demo_rolling_vs_expanding()
    bt = demo_model_comparison()
    demo_multi_horizon()
    demo_leakage_warning()

    # Plot
    try:
        fig = plot_backtest_results(bt, data)
        fig.savefig('.claude/backtest_demo_plot.png', dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to .claude/backtest_demo_plot.png")
        plt.close(fig)
    except Exception as e:
        print(f"\nPlotting skipped: {e}")

    print("\n" + "="*60)
    print("Demo completed successfully!")
    print("="*60)

    return bt


if __name__ == "__main__":
    run_demo()
