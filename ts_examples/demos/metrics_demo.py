"""
Forecast Accuracy Metrics Demo
==============================
Demonstrates various metrics for evaluating time series forecasts.

Related notes:
- docs/en/model-selection/cross-validation.md
- docs/en/forecasting/prediction-intervals.md
"""

import numpy as np
import matplotlib.pyplot as plt


def mae(actual, predicted):
    """
    Mean Absolute Error.

    MAE = (1/n) * Σ|y_t - ŷ_t|

    Properties:
    - Same units as the data
    - Robust to outliers (compared to MSE)
    - Easy to interpret
    """
    return np.mean(np.abs(actual - predicted))


def mse(actual, predicted):
    """
    Mean Squared Error.

    MSE = (1/n) * Σ(y_t - ŷ_t)²

    Properties:
    - Squared units
    - Penalizes large errors more heavily
    - Differentiable (useful for optimization)
    """
    return np.mean((actual - predicted)**2)


def rmse(actual, predicted):
    """
    Root Mean Squared Error.

    RMSE = √MSE

    Properties:
    - Same units as the data
    - More sensitive to outliers than MAE
    - Standard deviation of residuals (if unbiased)
    """
    return np.sqrt(mse(actual, predicted))


def mape(actual, predicted):
    """
    Mean Absolute Percentage Error.

    MAPE = (100/n) * Σ|y_t - ŷ_t|/|y_t|

    Properties:
    - Scale-independent (percentage)
    - Undefined when actual = 0
    - Asymmetric: penalizes over-forecasts less than under-forecasts
    """
    mask = actual != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def smape(actual, predicted):
    """
    Symmetric Mean Absolute Percentage Error.

    sMAPE = (100/n) * Σ|y_t - ŷ_t|/((|y_t| + |ŷ_t|)/2)

    Properties:
    - Bounded between 0% and 200%
    - More symmetric than MAPE
    - Still has issues when both actual and predicted are near zero
    """
    denominator = (np.abs(actual) + np.abs(predicted)) / 2
    mask = denominator != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs(actual[mask] - predicted[mask]) / denominator[mask]) * 100


def mase(actual, predicted, training_data, seasonality=1):
    """
    Mean Absolute Scaled Error.

    MASE = MAE / MAE_naive

    where MAE_naive is the MAE of a naive (seasonal) forecast on training data.

    Properties:
    - Scale-independent
    - Symmetric
    - MASE < 1 means better than naive forecast
    - Handles zero values properly
    """
    # Naive forecast MAE on training data
    if seasonality == 1:
        naive_errors = np.abs(training_data[1:] - training_data[:-1])
    else:
        naive_errors = np.abs(training_data[seasonality:] - training_data[:-seasonality])

    mae_naive = np.mean(naive_errors)

    if mae_naive == 0:
        return np.nan

    return mae(actual, predicted) / mae_naive


def wape(actual, predicted):
    """
    Weighted Absolute Percentage Error.

    WAPE = Σ|y_t - ŷ_t| / Σ|y_t|

    Properties:
    - Handles zeros better than MAPE
    - Gives more weight to larger values
    - Used in retail forecasting
    """
    sum_actual = np.sum(np.abs(actual))
    if sum_actual == 0:
        return np.nan
    return np.sum(np.abs(actual - predicted)) / sum_actual * 100


def coverage(actual, lower, upper):
    """
    Prediction interval coverage.

    Coverage = (number of actuals within interval) / n

    Target is typically 80%, 90%, or 95% depending on interval.
    """
    within = (actual >= lower) & (actual <= upper)
    return np.mean(within) * 100


def interval_score(actual, lower, upper, alpha=0.1):
    """
    Interval Score (Winkler Score).

    Proper scoring rule for prediction intervals.
    Lower is better.

    IS = (U - L) + (2/α)(L - y) * I(y < L) + (2/α)(y - U) * I(y > U)

    where α is 1 - confidence level (e.g., 0.1 for 90% interval)
    """
    width = upper - lower

    below = actual < lower
    above = actual > upper

    penalty_below = (2/alpha) * (lower - actual) * below
    penalty_above = (2/alpha) * (actual - upper) * above

    return np.mean(width + penalty_below + penalty_above)


def generate_forecast_data(n=100, seed=42):
    """Generate actual and forecast data for demonstration."""
    np.random.seed(seed)

    # Actual values (with some pattern)
    t = np.arange(n)
    actual = 100 + 0.5 * t + 10 * np.sin(2 * np.pi * t / 12) + np.random.randn(n) * 5

    # Good forecast (captures pattern + some noise)
    good_forecast = actual + np.random.randn(n) * 3

    # Biased forecast (systematic under-prediction)
    biased_forecast = actual - 8 + np.random.randn(n) * 3

    # Poor forecast (misses pattern)
    poor_forecast = 100 + np.random.randn(n) * 15

    # Training data (for MASE)
    training = 100 + 0.5 * np.arange(-50, 0) + 10 * np.sin(2 * np.pi * np.arange(-50, 0) / 12)

    return actual, good_forecast, biased_forecast, poor_forecast, training


def demo_basic_metrics():
    """Demonstrate basic accuracy metrics."""
    print("\n" + "="*50)
    print("Basic Accuracy Metrics")
    print("="*50)

    actual, good_fc, biased_fc, poor_fc, training = generate_forecast_data()

    print("\nMetric Comparison:")
    print(f"{'Metric':<15} {'Good FC':<12} {'Biased FC':<12} {'Poor FC':<12}")
    print("-"*51)

    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE (%)': mape,
        'SMAPE (%)': smape,
        'WAPE (%)': wape
    }

    for name, metric_fn in metrics.items():
        good_val = metric_fn(actual, good_fc)
        biased_val = metric_fn(actual, biased_fc)
        poor_val = metric_fn(actual, poor_fc)
        print(f"{name:<15} {good_val:<12.4f} {biased_val:<12.4f} {poor_val:<12.4f}")

    # MASE requires training data
    mase_good = mase(actual, good_fc, training, seasonality=12)
    mase_biased = mase(actual, biased_fc, training, seasonality=12)
    mase_poor = mase(actual, poor_fc, training, seasonality=12)
    print(f"{'MASE':<15} {mase_good:<12.4f} {mase_biased:<12.4f} {mase_poor:<12.4f}")

    print("\nInterpretation:")
    print("  - MASE < 1: Better than seasonal naive forecast")
    print("  - MAPE can be misleading with near-zero values")
    print("  - RMSE penalizes large errors more than MAE")


def demo_mape_issues():
    """Demonstrate MAPE asymmetry and zero issues."""
    print("\n" + "="*50)
    print("MAPE Issues Demonstration")
    print("="*50)

    # Asymmetry example
    actual = np.array([100, 100, 100])
    over_forecast = np.array([150, 150, 150])  # +50% error
    under_forecast = np.array([50, 50, 50])    # -50% error

    print("\nAsymmetry Issue:")
    print(f"  Actual: {actual}")
    print(f"  Over-forecast (+50): {over_forecast}, MAPE = {mape(actual, over_forecast):.1f}%")
    print(f"  Under-forecast (-50): {under_forecast}, MAPE = {mape(actual, under_forecast):.1f}%")
    print("  → Same absolute error, different MAPE!")

    # Zero issue
    actual_with_zero = np.array([0, 50, 100])
    forecast_for_zero = np.array([5, 55, 105])

    print("\nZero Value Issue:")
    print(f"  Actual: {actual_with_zero}")
    print(f"  Forecast: {forecast_for_zero}")
    print(f"  MAPE (excluding zeros): {mape(actual_with_zero, forecast_for_zero):.1f}%")
    print("  → MAPE is undefined when actual = 0")

    # SMAPE comparison
    print("\nSMAPE as Alternative:")
    print(f"  sMAPE (over-forecast): {smape(actual, over_forecast):.1f}%")
    print(f"  sMAPE (under-forecast): {smape(actual, under_forecast):.1f}%")
    print("  → More symmetric but still has issues near zero")


def demo_scale_dependence():
    """Demonstrate scale-dependent vs scale-independent metrics."""
    print("\n" + "="*50)
    print("Scale Dependence Demonstration")
    print("="*50)

    np.random.seed(42)

    # Two series at different scales
    actual_small = 10 + np.random.randn(50) * 2
    actual_large = 1000 + np.random.randn(50) * 200

    # Same relative forecast quality
    forecast_small = actual_small + np.random.randn(50) * 1
    forecast_large = actual_large + np.random.randn(50) * 100

    print("\nScale-Dependent Metrics:")
    print(f"  Small scale MAE:  {mae(actual_small, forecast_small):.4f}")
    print(f"  Large scale MAE:  {mae(actual_large, forecast_large):.4f}")
    print(f"  Small scale RMSE: {rmse(actual_small, forecast_small):.4f}")
    print(f"  Large scale RMSE: {rmse(actual_large, forecast_large):.4f}")

    print("\nScale-Independent Metrics:")
    print(f"  Small scale MAPE: {mape(actual_small, forecast_small):.2f}%")
    print(f"  Large scale MAPE: {mape(actual_large, forecast_large):.2f}%")

    # Generate training data for MASE
    training_small = 10 + np.random.randn(50) * 2
    training_large = 1000 + np.random.randn(50) * 200

    print(f"  Small scale MASE: {mase(actual_small, forecast_small, training_small):.4f}")
    print(f"  Large scale MASE: {mase(actual_large, forecast_large, training_large):.4f}")
    print("\n  → MASE and MAPE allow cross-series comparison")


def demo_prediction_intervals():
    """Demonstrate prediction interval evaluation."""
    print("\n" + "="*50)
    print("Prediction Interval Evaluation")
    print("="*50)

    np.random.seed(42)
    n = 100

    # Actual values
    actual = 100 + np.random.randn(n) * 10

    # Point forecast
    forecast = 100 + np.random.randn(n) * 2

    # Prediction intervals (90%)
    # Well-calibrated interval
    good_lower = forecast - 16.5  # ~90% coverage for N(0,10)
    good_upper = forecast + 16.5

    # Too narrow interval
    narrow_lower = forecast - 5
    narrow_upper = forecast + 5

    # Too wide interval
    wide_lower = forecast - 30
    wide_upper = forecast + 30

    print("\n90% Prediction Interval Evaluation:")
    print(f"{'Interval':<15} {'Coverage':<12} {'Avg Width':<12} {'Int. Score':<12}")
    print("-"*51)

    for name, lower, upper in [
        ('Well-calibrated', good_lower, good_upper),
        ('Too narrow', narrow_lower, narrow_upper),
        ('Too wide', wide_lower, wide_upper)
    ]:
        cov = coverage(actual, lower, upper)
        width = np.mean(upper - lower)
        score = interval_score(actual, lower, upper, alpha=0.1)
        print(f"{name:<15} {cov:<12.1f} {width:<12.2f} {score:<12.2f}")

    print("\nKey insights:")
    print("  - Target coverage: 90%")
    print("  - Interval score rewards narrow intervals with correct coverage")
    print("  - Under-coverage is heavily penalized")


def demo_metric_selection_guide():
    """Provide guidance on metric selection."""
    print("\n" + "="*50)
    print("Metric Selection Guide")
    print("="*50)

    print("""
    When to use each metric:

    MAE:
    - Default choice for point forecasts
    - Interpretable, same units as data
    - Use when outlier robustness matters

    RMSE:
    - When large errors are especially costly
    - Standard in many academic contexts
    - Differentiable (good for optimization)

    MAPE:
    - When relative errors matter
    - When comparing across different scales
    - AVOID with near-zero or intermittent data

    SMAPE:
    - Better than MAPE for symmetry
    - Bounded 0-200%
    - Still problematic near zero

    MASE:
    - Best scale-independent metric
    - Handles zeros properly
    - Interpretable: < 1 means better than naive
    - Requires training data

    WAPE:
    - Good for aggregated forecast accuracy
    - Common in retail/demand forecasting
    - Handles zeros better than MAPE

    Interval Score:
    - For prediction intervals
    - Proper scoring rule
    - Rewards calibration and sharpness
    """)


def plot_metric_comparison(actual, forecasts, names, filename='.claude/metrics_demo_plot.png'):
    """Plot forecast comparison with metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Time series plot
    t = np.arange(len(actual))
    axes[0, 0].plot(t, actual, 'k-', linewidth=2, label='Actual')
    for fc, name in zip(forecasts, names):
        axes[0, 0].plot(t, fc, '--', alpha=0.7, label=name)
    axes[0, 0].set_title('Forecasts vs Actual')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Error distributions
    for i, (fc, name) in enumerate(zip(forecasts, names)):
        errors = actual - fc
        axes[0, 1].hist(errors, bins=20, alpha=0.5, label=name)
    axes[0, 1].axvline(x=0, color='k', linestyle='--')
    axes[0, 1].set_title('Error Distributions')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Metric comparison bar chart
    metric_names = ['MAE', 'RMSE', 'MAPE', 'SMAPE']
    x = np.arange(len(metric_names))
    width = 0.25

    for i, (fc, name) in enumerate(zip(forecasts, names)):
        values = [
            mae(actual, fc),
            rmse(actual, fc),
            mape(actual, fc),
            smape(actual, fc)
        ]
        axes[1, 0].bar(x + i*width, values, width, label=name)

    axes[1, 0].set_xticks(x + width)
    axes[1, 0].set_xticklabels(metric_names)
    axes[1, 0].set_title('Metric Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Scatter plot
    axes[1, 1].scatter(actual, forecasts[0], alpha=0.5, label=names[0])
    lims = [min(actual.min(), forecasts[0].min()), max(actual.max(), forecasts[0].max())]
    axes[1, 1].plot(lims, lims, 'r--', label='Perfect')
    axes[1, 1].set_xlabel('Actual')
    axes[1, 1].set_ylabel('Forecast')
    axes[1, 1].set_title(f'Actual vs Forecast ({names[0]})')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def run_demo():
    """Run the complete metrics demo."""
    print("="*60)
    print("Forecast Accuracy Metrics Demonstration")
    print("="*60)

    # Run all demos
    demo_basic_metrics()
    demo_mape_issues()
    demo_scale_dependence()
    demo_prediction_intervals()
    demo_metric_selection_guide()

    # Plot
    actual, good_fc, biased_fc, poor_fc, _ = generate_forecast_data()
    try:
        fig = plot_metric_comparison(
            actual,
            [good_fc, biased_fc, poor_fc],
            ['Good', 'Biased', 'Poor']
        )
        fig.savefig('.claude/metrics_demo_plot.png', dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to .claude/metrics_demo_plot.png")
        plt.close(fig)
    except Exception as e:
        print(f"\nPlotting skipped: {e}")

    print("\n" + "="*60)
    print("Demo completed successfully!")
    print("="*60)


if __name__ == "__main__":
    run_demo()
