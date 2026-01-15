"""
Vector Autoregression (VAR) Demo
================================
Demonstrates VAR model fitting and Granger causality testing.

Related notes:
- docs/en/multivariate/var.md
- docs/en/multivariate/granger-causality.md
"""

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests, adfuller


def generate_var_data(n=300, seed=42):
    """
    Generate VAR(2) data with known coefficients.

    System:
    y1_t = 0.5*y1_{t-1} - 0.3*y1_{t-2} + 0.2*y2_{t-1} + eps1
    y2_t = 0.3*y1_{t-1} + 0.4*y2_{t-1} - 0.2*y2_{t-2} + eps2
    """
    np.random.seed(seed)

    # Coefficient matrices
    A1 = np.array([[0.5, 0.2],
                   [0.3, 0.4]])
    A2 = np.array([[-0.3, 0.0],
                   [0.0, -0.2]])

    # Generate data
    y = np.zeros((n, 2))
    eps = np.random.randn(n, 2) * np.array([1.0, 0.8])

    for t in range(2, n):
        y[t] = A1 @ y[t-1] + A2 @ y[t-2] + eps[t]

    return y, A1, A2


def check_stationarity(data, names=['y1', 'y2']):
    """Check stationarity of each series."""
    print("\nStationarity Tests (ADF):")
    print("-"*40)
    for i, name in enumerate(names):
        result = adfuller(data[:, i])
        status = 'Stationary' if result[1] < 0.05 else 'Non-stationary'
        print(f"{name}: p-value = {result[1]:.4f} → {status}")


def fit_var_model(data):
    """Fit VAR model with order selection."""
    print("\n" + "="*50)
    print("VAR Model Fitting")
    print("="*50)

    model = VAR(data)

    # Order selection
    print("\nOrder Selection (Information Criteria):")
    print(f"{'Lag':<6} {'AIC':<12} {'BIC':<12} {'HQIC':<12}")
    print("-"*42)

    for p in range(1, 6):
        result = model.fit(p)
        print(f"{p:<6} {result.aic:<12.2f} {result.bic:<12.2f} {result.hqic:<12.2f}")

    # Select by AIC
    result = model.fit(maxlags=5, ic='aic')
    print(f"\nSelected order (AIC): {result.k_ar}")

    # Display coefficients
    print("\nEstimated Coefficients:")
    for i, eq in enumerate(['y1', 'y2']):
        print(f"\n{eq} equation:")
        params = result.params[eq]
        for name, val in params.items():
            print(f"  {name}: {val:.4f}")

    return result


def test_granger_causality(data, maxlag=4):
    """Test Granger causality in both directions."""
    print("\n" + "="*50)
    print("Granger Causality Tests")
    print("="*50)

    print(f"\nTesting: y2 Granger-causes y1")
    print("-"*40)
    gc_y2_to_y1 = grangercausalitytests(data[:, [0, 1]], maxlag=maxlag, verbose=False)
    for lag, result in gc_y2_to_y1.items():
        f_stat = result[0]['ssr_ftest'][0]
        p_val = result[0]['ssr_ftest'][1]
        sig = '*' if p_val < 0.05 else ''
        print(f"  Lag {lag}: F={f_stat:.2f}, p-value={p_val:.4f} {sig}")

    print(f"\nTesting: y1 Granger-causes y2")
    print("-"*40)
    gc_y1_to_y2 = grangercausalitytests(data[:, [1, 0]], maxlag=maxlag, verbose=False)
    for lag, result in gc_y1_to_y2.items():
        f_stat = result[0]['ssr_ftest'][0]
        p_val = result[0]['ssr_ftest'][1]
        sig = '*' if p_val < 0.05 else ''
        print(f"  Lag {lag}: F={f_stat:.2f}, p-value={p_val:.4f} {sig}")

    return gc_y2_to_y1, gc_y1_to_y2


def compute_irf(result, periods=20):
    """Compute and display impulse response functions."""
    print("\n" + "="*50)
    print("Impulse Response Analysis")
    print("="*50)

    irf = result.irf(periods)

    print("\nImpulse Response: Shock to y1")
    print(f"{'Period':<8} {'y1 response':<15} {'y2 response':<15}")
    print("-"*38)
    for i in [0, 1, 2, 5, 10, 20]:
        if i < periods:
            print(f"{i:<8} {irf.irfs[i, 0, 0]:<15.4f} {irf.irfs[i, 1, 0]:<15.4f}")

    print("\nImpulse Response: Shock to y2")
    print(f"{'Period':<8} {'y1 response':<15} {'y2 response':<15}")
    print("-"*38)
    for i in [0, 1, 2, 5, 10, 20]:
        if i < periods:
            print(f"{i:<8} {irf.irfs[i, 0, 1]:<15.4f} {irf.irfs[i, 1, 1]:<15.4f}")

    return irf


def forecast_var(result, steps=20):
    """Generate and display forecasts."""
    print("\n" + "="*50)
    print("VAR Forecasting")
    print("="*50)

    forecast = result.forecast(result.endog[-result.k_ar:], steps)

    print(f"\n{steps}-step ahead forecasts:")
    print(f"{'Step':<8} {'y1':<15} {'y2':<15}")
    print("-"*38)
    for i in [0, 4, 9, 14, 19]:
        if i < steps:
            print(f"{i+1:<8} {forecast[i, 0]:<15.4f} {forecast[i, 1]:<15.4f}")

    return forecast


def plot_results(data, result, filename='.claude/var_demo_plot.png'):
    """Plot VAR results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Time series
    axes[0, 0].plot(data[:, 0], label='y1')
    axes[0, 0].plot(data[:, 1], label='y2')
    axes[0, 0].set_title('Original Time Series')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Fitted values
    fitted = result.fittedvalues
    axes[0, 1].plot(data[result.k_ar:, 0], 'b-', alpha=0.5, label='y1 actual')
    axes[0, 1].plot(fitted[:, 0], 'b--', label='y1 fitted')
    axes[0, 1].plot(data[result.k_ar:, 1], 'r-', alpha=0.5, label='y2 actual')
    axes[0, 1].plot(fitted[:, 1], 'r--', label='y2 fitted')
    axes[0, 1].set_title('Fitted Values')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # IRF for shock to y1
    irf = result.irf(20)
    axes[1, 0].plot(irf.irfs[:, 0, 0], 'b-', label='y1 → y1')
    axes[1, 0].plot(irf.irfs[:, 1, 0], 'r-', label='y1 → y2')
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 0].set_title('IRF: Shock to y1')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # IRF for shock to y2
    axes[1, 1].plot(irf.irfs[:, 0, 1], 'b-', label='y2 → y1')
    axes[1, 1].plot(irf.irfs[:, 1, 1], 'r-', label='y2 → y2')
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 1].set_title('IRF: Shock to y2')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def run_demo():
    """Run the complete VAR demo."""
    print("="*60)
    print("Vector Autoregression (VAR) Demonstration")
    print("="*60)

    # Generate data
    print("\n[1] Generating VAR(2) data...")
    data, A1, A2 = generate_var_data()
    print(f"    Generated {len(data)} observations")
    print(f"    True A1:\n{A1}")
    print(f"    True A2:\n{A2}")

    # Check stationarity
    print("\n[2] Checking stationarity...")
    check_stationarity(data)

    # Fit model
    print("\n[3] Fitting VAR model...")
    result = fit_var_model(data)

    # Granger causality
    print("\n[4] Testing Granger causality...")
    test_granger_causality(data)

    # IRF analysis
    print("\n[5] Computing impulse responses...")
    irf = compute_irf(result)

    # Forecasting
    print("\n[6] Generating forecasts...")
    forecast = forecast_var(result)

    # Plot
    try:
        fig = plot_results(data, result)
        fig.savefig('.claude/var_demo_plot.png', dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to .claude/var_demo_plot.png")
        plt.close(fig)
    except Exception as e:
        print(f"\nPlotting skipped: {e}")

    print("\n" + "="*60)
    print("Demo completed successfully!")
    print("="*60)

    return result


if __name__ == "__main__":
    run_demo()
