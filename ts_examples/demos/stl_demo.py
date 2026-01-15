"""
STL Decomposition Demo
======================
Demonstrates Seasonal and Trend decomposition using Loess.

Related notes:
- docs/en/decomposition/stl.md
- docs/en/decomposition/classical.md
"""

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL, seasonal_decompose


def generate_seasonal_data(n=144, period=12, seed=42):
    """Generate data with trend, seasonality, and noise."""
    np.random.seed(seed)
    t = np.arange(n)

    # Trend (slowly varying)
    trend = 50 + 0.1 * t + 5 * np.sin(2 * np.pi * t / (n/2))

    # Seasonality (period=12)
    seasonal = 10 * np.sin(2 * np.pi * t / period) + 5 * np.cos(4 * np.pi * t / period)

    # Noise
    noise = np.random.randn(n) * 3

    # Add some outliers
    outlier_idx = [25, 80, 120]
    noise[outlier_idx] += np.array([20, -25, 15])

    return trend + seasonal + noise, trend, seasonal


def demo_stl_decomposition():
    """Demonstrate STL decomposition."""
    print("\n" + "="*50)
    print("STL Decomposition")
    print("="*50)

    # Generate data
    y, true_trend, true_seasonal = generate_seasonal_data()

    # Perform STL decomposition
    stl = STL(y, period=12, robust=True)
    result = stl.fit()

    print("\nSTL Components:")
    print(f"  Trend range:      [{result.trend.min():.2f}, {result.trend.max():.2f}]")
    print(f"  Seasonal range:   [{result.seasonal.min():.2f}, {result.seasonal.max():.2f}]")
    print(f"  Residual std:     {result.resid.std():.4f}")

    # Check decomposition quality
    reconstruction = result.trend + result.seasonal + result.resid
    reconstruction_error = np.mean(np.abs(y - reconstruction))
    print(f"  Reconstruction error: {reconstruction_error:.6f}")

    # Compare trend estimation
    trend_corr = np.corrcoef(result.trend, true_trend)[0, 1]
    seasonal_corr = np.corrcoef(result.seasonal, true_seasonal)[0, 1]
    print(f"\nCorrelation with true components:")
    print(f"  Trend correlation:    {trend_corr:.4f}")
    print(f"  Seasonal correlation: {seasonal_corr:.4f}")

    return result, y


def demo_robust_vs_nonrobust():
    """Compare robust vs non-robust STL."""
    print("\n" + "="*50)
    print("Robust vs Non-Robust STL")
    print("="*50)

    # Generate data with outliers
    y, _, _ = generate_seasonal_data()

    # Non-robust STL
    stl_nonrobust = STL(y, period=12, robust=False)
    result_nonrobust = stl_nonrobust.fit()

    # Robust STL
    stl_robust = STL(y, period=12, robust=True)
    result_robust = stl_robust.fit()

    # Compare residuals
    print("\nResidual Statistics:")
    print(f"{'Metric':<20} {'Non-Robust':<15} {'Robust':<15}")
    print("-"*50)
    print(f"{'Std':<20} {result_nonrobust.resid.std():<15.4f} {result_robust.resid.std():<15.4f}")
    print(f"{'Max abs':<20} {np.abs(result_nonrobust.resid).max():<15.4f} {np.abs(result_robust.resid).max():<15.4f}")
    print(f"{'MAD':<20} {np.median(np.abs(result_nonrobust.resid)):<15.4f} {np.median(np.abs(result_robust.resid)):<15.4f}")

    # Check outlier detection via weights
    weights = result_robust.weights
    outlier_count = np.sum(weights < 0.5)
    print(f"\nDetected outliers (weight < 0.5): {outlier_count}")
    print(f"Outlier indices: {np.where(weights < 0.5)[0]}")

    return result_nonrobust, result_robust, y


def demo_classical_vs_stl():
    """Compare classical decomposition with STL."""
    print("\n" + "="*50)
    print("Classical Decomposition vs STL")
    print("="*50)

    # Generate data
    y, true_trend, true_seasonal = generate_seasonal_data()

    # Classical decomposition (additive)
    classical_result = seasonal_decompose(y, model='additive', period=12)

    # STL decomposition
    stl = STL(y, period=12, robust=True)
    stl_result = stl.fit()

    # Compare
    print("\nTrend Comparison:")
    classical_trend_corr = np.corrcoef(
        classical_result.trend[~np.isnan(classical_result.trend)],
        true_trend[~np.isnan(classical_result.trend)]
    )[0, 1]
    stl_trend_corr = np.corrcoef(stl_result.trend, true_trend)[0, 1]
    print(f"  Classical correlation with true: {classical_trend_corr:.4f}")
    print(f"  STL correlation with true:       {stl_trend_corr:.4f}")

    print("\nSeasonal Comparison:")
    classical_seasonal_corr = np.corrcoef(
        classical_result.seasonal[~np.isnan(classical_result.seasonal)],
        true_seasonal[~np.isnan(classical_result.seasonal)]
    )[0, 1]
    stl_seasonal_corr = np.corrcoef(stl_result.seasonal, true_seasonal)[0, 1]
    print(f"  Classical correlation with true: {classical_seasonal_corr:.4f}")
    print(f"  STL correlation with true:       {stl_seasonal_corr:.4f}")

    print("\nKey Differences:")
    print("  - STL: Handles all time points, robust to outliers")
    print("  - Classical: Loses endpoints, no outlier robustness")

    return classical_result, stl_result, y


def plot_decomposition(result, y, filename='.claude/stl_demo_plot.png'):
    """Plot STL decomposition results."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))

    axes[0].plot(y, 'b-', label='Original')
    axes[0].set_title('Original Series')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(result.trend, 'g-', label='Trend')
    axes[1].set_title('Trend Component')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(result.seasonal, 'orange', label='Seasonal')
    axes[2].set_title('Seasonal Component')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(result.resid, 'r-', label='Residual', alpha=0.7)
    axes[3].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[3].set_title('Residual Component')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def run_demo():
    """Run the complete STL demo."""
    print("="*60)
    print("STL Decomposition Demonstration")
    print("="*60)

    # Run all demos
    stl_result, y = demo_stl_decomposition()
    nonrobust, robust, _ = demo_robust_vs_nonrobust()
    classical, stl, _ = demo_classical_vs_stl()

    # Plot
    try:
        fig = plot_decomposition(stl_result, y)
        fig.savefig('.claude/stl_demo_plot.png', dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to .claude/stl_demo_plot.png")
        plt.close(fig)
    except Exception as e:
        print(f"\nPlotting skipped: {e}")

    print("\n" + "="*60)
    print("Demo completed successfully!")
    print("="*60)

    return stl_result


if __name__ == "__main__":
    run_demo()
