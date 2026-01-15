"""
ETS / Exponential Smoothing Demo
================================
Demonstrates SES, Holt, and Holt-Winters methods.

Related notes:
- docs/en/exponential-smoothing/ses.md
- docs/en/exponential-smoothing/holt.md
- docs/en/exponential-smoothing/holt-winters.md
- docs/en/exponential-smoothing/ets.md
"""

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing


def generate_level_data(n=100, level=50, noise_std=5, seed=42):
    """Generate data with constant level + noise (for SES)."""
    np.random.seed(seed)
    return level + np.random.randn(n) * noise_std


def generate_trend_data(n=100, level=50, trend=0.5, noise_std=5, seed=42):
    """Generate data with linear trend + noise (for Holt)."""
    np.random.seed(seed)
    t = np.arange(n)
    return level + trend * t + np.random.randn(n) * noise_std


def generate_seasonal_data(n=120, level=50, trend=0.3, seasonal_amplitude=15,
                          period=12, noise_std=3, seed=42):
    """Generate data with trend + seasonality + noise (for Holt-Winters)."""
    np.random.seed(seed)
    t = np.arange(n)
    seasonal = seasonal_amplitude * np.sin(2 * np.pi * t / period)
    return level + trend * t + seasonal + np.random.randn(n) * noise_std


def demo_ses():
    """Demonstrate Simple Exponential Smoothing."""
    print("\n" + "="*50)
    print("Simple Exponential Smoothing (SES)")
    print("="*50)

    # Generate level data
    y = generate_level_data()

    # Fit with optimization
    model = SimpleExpSmoothing(y, initialization_method='estimated')
    fit = model.fit(optimized=True)

    print(f"\nOptimal α: {fit.params['smoothing_level']:.4f}")
    print(f"Initial level: {fit.params['initial_level']:.2f}")

    # Compare different α values
    print("\nComparison of different α values:")
    print(f"{'Alpha':<10} {'SSE':<15} {'RMSE':<15}")
    print("-"*40)

    for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
        fit_alpha = model.fit(smoothing_level=alpha, optimized=False)
        residuals = y - fit_alpha.fittedvalues
        sse = np.sum(residuals**2)
        rmse = np.sqrt(np.mean(residuals**2))
        print(f"{alpha:<10} {sse:<15.2f} {rmse:<15.4f}")

    # Forecast
    forecast = fit.forecast(10)
    print(f"\n10-step forecast (all same for SES): {forecast.values[0]:.2f}")

    return fit, y


def demo_holt():
    """Demonstrate Holt's Linear Method."""
    print("\n" + "="*50)
    print("Holt's Linear Method")
    print("="*50)

    # Generate trend data
    y = generate_trend_data()

    # Fit Holt's method (no damping)
    model = ExponentialSmoothing(y, trend='add', seasonal=None,
                                 initialization_method='estimated')
    fit = model.fit(optimized=True)

    print(f"\nOptimal α (level): {fit.params['smoothing_level']:.4f}")
    print(f"Optimal β (trend): {fit.params['smoothing_trend']:.4f}")
    print(f"Initial level: {fit.params['initial_level']:.2f}")
    print(f"Initial trend: {fit.params['initial_trend']:.4f}")

    # Compare with damped trend
    model_damped = ExponentialSmoothing(y, trend='add', damped_trend=True,
                                        seasonal=None, initialization_method='estimated')
    fit_damped = model_damped.fit(optimized=True)

    print(f"\nDamped trend:")
    print(f"  φ (damping): {fit_damped.params['damping_trend']:.4f}")

    # Forecast comparison
    h = 20
    fc_holt = fit.forecast(h)
    fc_damped = fit_damped.forecast(h)

    print(f"\n{h}-step forecast comparison:")
    print(f"  Holt (undamped): {fc_holt.values[-1]:.2f}")
    print(f"  Holt (damped):   {fc_damped.values[-1]:.2f}")
    print(f"  Difference:      {fc_holt.values[-1] - fc_damped.values[-1]:.2f}")

    return fit, y


def demo_holt_winters():
    """Demonstrate Holt-Winters Method."""
    print("\n" + "="*50)
    print("Holt-Winters Seasonal Method")
    print("="*50)

    # Generate seasonal data
    y = generate_seasonal_data(n=120, period=12)

    # Fit additive Holt-Winters
    model_add = ExponentialSmoothing(y, trend='add', seasonal='add',
                                     seasonal_periods=12,
                                     initialization_method='estimated')
    fit_add = model_add.fit(optimized=True)

    print("\nAdditive Holt-Winters:")
    print(f"  α (level):    {fit_add.params['smoothing_level']:.4f}")
    print(f"  β (trend):    {fit_add.params['smoothing_trend']:.4f}")
    print(f"  γ (seasonal): {fit_add.params['smoothing_seasonal']:.4f}")

    # Fit multiplicative Holt-Winters
    y_pos = y - y.min() + 10  # Ensure positive values
    model_mul = ExponentialSmoothing(y_pos, trend='add', seasonal='mul',
                                     seasonal_periods=12,
                                     initialization_method='estimated')
    fit_mul = model_mul.fit(optimized=True)

    print("\nMultiplicative Holt-Winters:")
    print(f"  α (level):    {fit_mul.params['smoothing_level']:.4f}")
    print(f"  β (trend):    {fit_mul.params['smoothing_trend']:.4f}")
    print(f"  γ (seasonal): {fit_mul.params['smoothing_seasonal']:.4f}")

    # Model comparison
    sse_add = np.sum((y - fit_add.fittedvalues)**2)
    sse_mul = np.sum((y_pos - fit_mul.fittedvalues)**2)

    print("\nModel Comparison (SSE):")
    print(f"  Additive:       {sse_add:.2f}")
    print(f"  Multiplicative: {sse_mul:.2f} (on shifted data)")

    # Forecast
    forecast = fit_add.forecast(24)
    print(f"\n24-step forecast range: [{forecast.min():.2f}, {forecast.max():.2f}]")

    return fit_add, y


def plot_results(fits, series_list, names, filename='.claude/ets_demo_plot.png'):
    """Plot all ETS results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, fit, series, name in zip(axes, fits, series_list, names):
        n = len(series)
        h = 20 if 'Holt-Winters' not in name else 24
        forecast = fit.forecast(h)

        # Plot
        ax.plot(range(n), series, 'b-', label='Data', alpha=0.7)
        ax.plot(range(n), fit.fittedvalues, 'g--', label='Fitted', alpha=0.5)
        ax.plot(range(n, n+h), forecast, 'r-', label='Forecast', linewidth=2)

        ax.set_title(name)
        ax.set_xlabel('Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def run_demo():
    """Run the complete ETS demo."""
    print("="*60)
    print("Exponential Smoothing (ETS) Demonstration")
    print("="*60)

    # Run all demos
    fit_ses, y_ses = demo_ses()
    fit_holt, y_holt = demo_holt()
    fit_hw, y_hw = demo_holt_winters()

    # Plot
    try:
        fig = plot_results(
            [fit_ses, fit_holt, fit_hw],
            [y_ses, y_holt, y_hw],
            ['SES', 'Holt', 'Holt-Winters']
        )
        fig.savefig('.claude/ets_demo_plot.png', dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to .claude/ets_demo_plot.png")
        plt.close(fig)
    except Exception as e:
        print(f"\nPlotting skipped: {e}")

    print("\n" + "="*60)
    print("Demo completed successfully!")
    print("="*60)

    return fit_ses, fit_holt, fit_hw


if __name__ == "__main__":
    run_demo()
