"""
ARIMA Model Demo
================
Demonstrates ARIMA model fitting, diagnostics, and forecasting.

Related notes:
- docs/en/time-domain/arima.md
- docs/en/time-domain/identification.md
"""

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox


def generate_arima_data(n=300, phi=0.7, theta=0.4, d=1, seed=42):
    """Generate ARIMA(1,d,1) data."""
    np.random.seed(seed)
    eps = np.random.randn(n + 10)

    # Generate ARMA(1,1) for the differenced series
    diff_series = np.zeros(n)
    diff_series[0] = eps[1] + theta * eps[0]
    for t in range(1, n):
        diff_series[t] = phi * diff_series[t-1] + eps[t+1] + theta * eps[t]

    # Integrate d times
    series = diff_series.copy()
    for _ in range(d):
        series = np.cumsum(series)

    return series + 100  # Add level


def test_stationarity(series, name="Series"):
    """Run ADF test and print results."""
    result = adfuller(series, autolag='AIC')
    print(f"\n{name} - ADF Test:")
    print(f"  ADF Statistic: {result[0]:.4f}")
    print(f"  p-value: {result[1]:.4f}")
    print(f"  Conclusion: {'Stationary' if result[1] < 0.05 else 'Non-stationary'}")
    return result[1] < 0.05


def fit_and_diagnose(series, order, verbose=True):
    """Fit ARIMA model and run diagnostics."""
    model = ARIMA(series, order=order)
    result = model.fit()

    if verbose:
        print(f"\n{'='*50}")
        print(f"ARIMA{order} Model Results")
        print(f"{'='*50}")
        print(f"AIC: {result.aic:.2f}")
        print(f"BIC: {result.bic:.2f}")
        print(f"\nParameters:")
        for name, val in zip(result.param_names, result.params):
            print(f"  {name}: {val:.4f}")

        # Ljung-Box test
        lb_test = acorr_ljungbox(result.resid, lags=[10], return_df=True)
        print(f"\nLjung-Box Test (lag 10):")
        print(f"  Q-statistic: {lb_test['lb_stat'].values[0]:.2f}")
        print(f"  p-value: {lb_test['lb_pvalue'].values[0]:.4f}")
        print(f"  Residuals white noise: {'Yes' if lb_test['lb_pvalue'].values[0] > 0.05 else 'No'}")

    return result


def compare_models(series, orders):
    """Compare multiple ARIMA specifications."""
    print("\n" + "="*60)
    print("Model Comparison")
    print("="*60)
    print(f"{'Model':<15} {'AIC':<12} {'BIC':<12} {'LB p-value':<12}")
    print("-"*60)

    results = {}
    for order in orders:
        try:
            model = ARIMA(series, order=order).fit()
            lb_test = acorr_ljungbox(model.resid, lags=[10], return_df=True)
            lb_pval = lb_test['lb_pvalue'].values[0]
            print(f"ARIMA{order:<10} {model.aic:<12.2f} {model.bic:<12.2f} {lb_pval:<12.4f}")
            results[order] = {'model': model, 'aic': model.aic, 'bic': model.bic}
        except Exception as e:
            print(f"ARIMA{order:<10} Failed: {str(e)[:30]}")

    return results


def forecast_and_plot(result, series, steps=20):
    """Generate forecasts with prediction intervals."""
    forecast = result.get_forecast(steps=steps)
    forecast_mean = forecast.predicted_mean
    conf_int = forecast.conf_int()

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Historical data
    ax.plot(range(len(series)), series, 'b-', label='Historical', alpha=0.7)

    # Fitted values
    ax.plot(range(len(series)), result.fittedvalues, 'g--', label='Fitted', alpha=0.5)

    # Forecasts
    forecast_idx = range(len(series), len(series) + steps)
    ax.plot(forecast_idx, forecast_mean, 'r-', label='Forecast', linewidth=2)
    ax.fill_between(forecast_idx,
                    conf_int.iloc[:, 0],
                    conf_int.iloc[:, 1],
                    color='red', alpha=0.2, label='95% CI')

    ax.legend()
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title(f'ARIMA{result.model.order} Forecast')
    ax.grid(True, alpha=0.3)

    return fig


def run_demo():
    """Run the complete ARIMA demo."""
    print("="*60)
    print("ARIMA Model Demonstration")
    print("="*60)

    # 1. Generate data
    print("\n[1] Generating ARIMA(1,1,1) data...")
    true_phi, true_theta = 0.7, 0.4
    series = generate_arima_data(phi=true_phi, theta=true_theta)
    print(f"    Generated {len(series)} observations")
    print(f"    True parameters: φ = {true_phi}, θ = {true_theta}")

    # 2. Test stationarity
    print("\n[2] Testing stationarity...")
    is_stationary = test_stationarity(series, "Original series")

    diff_series = np.diff(series)
    is_diff_stationary = test_stationarity(diff_series, "Differenced series")

    # 3. Examine ACF/PACF
    print("\n[3] ACF/PACF of differenced series:")
    acf_vals = acf(diff_series, nlags=10)
    pacf_vals = pacf(diff_series, nlags=10)
    print(f"    ACF (lags 1-5):  {np.round(acf_vals[1:6], 3)}")
    print(f"    PACF (lags 1-5): {np.round(pacf_vals[1:6], 3)}")

    # 4. Fit model
    print("\n[4] Fitting ARIMA(1,1,1)...")
    result = fit_and_diagnose(series, order=(1, 1, 1))

    # 5. Compare models
    print("\n[5] Comparing candidate models...")
    orders = [(1, 1, 0), (0, 1, 1), (1, 1, 1), (2, 1, 1), (1, 1, 2)]
    compare_models(series, orders)

    # 6. Forecast
    print("\n[6] Generating 20-step forecast...")
    forecast = result.get_forecast(20)
    print(f"    Forecast mean (first 5): {np.round(forecast.predicted_mean.values[:5], 2)}")

    # 7. Plot (save to file if not interactive)
    try:
        fig = forecast_and_plot(result, series)
        fig.savefig('.claude/arima_demo_plot.png', dpi=150, bbox_inches='tight')
        print(f"\n[7] Plot saved to .claude/arima_demo_plot.png")
        plt.close(fig)
    except Exception as e:
        print(f"\n[7] Plotting skipped: {e}")

    print("\n" + "="*60)
    print("Demo completed successfully!")
    print("="*60)

    return result


if __name__ == "__main__":
    run_demo()
