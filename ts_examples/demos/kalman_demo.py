"""
Kalman Filter Demo
==================
Demonstrates Kalman filter for local level and local linear trend models.

Related notes:
- docs/en/state-space/kalman-filter.md
"""

import numpy as np
import matplotlib.pyplot as plt


class KalmanFilter:
    """
    Simple Kalman filter for local level model.

    State space form:
        Observation: y_t = x_t + v_t,  v_t ~ N(0, R)
        State:       x_t = x_{t-1} + w_t,  w_t ~ N(0, Q)
    """

    def __init__(self, Q, R, x0=0, P0=1):
        """
        Initialize Kalman filter.

        Parameters:
        -----------
        Q : float - State noise variance
        R : float - Observation noise variance
        x0 : float - Initial state estimate
        P0 : float - Initial state variance
        """
        self.Q = Q  # State noise variance
        self.R = R  # Observation noise variance
        self.x = x0  # Current state estimate
        self.P = P0  # Current state variance

    def predict(self):
        """Prediction step."""
        # State prediction: x_pred = x (random walk)
        self.x_pred = self.x
        # Variance prediction: P_pred = P + Q
        self.P_pred = self.P + self.Q
        return self.x_pred, self.P_pred

    def update(self, y):
        """Update step with new observation."""
        # Kalman gain: K = P_pred / (P_pred + R)
        self.K = self.P_pred / (self.P_pred + self.R)

        # State update: x = x_pred + K * (y - x_pred)
        self.x = self.x_pred + self.K * (y - self.x_pred)

        # Variance update: P = (1 - K) * P_pred
        self.P = (1 - self.K) * self.P_pred

        return self.x, self.P, self.K

    def filter(self, observations):
        """Run filter over entire series."""
        n = len(observations)
        filtered_states = np.zeros(n)
        filtered_variances = np.zeros(n)
        kalman_gains = np.zeros(n)

        for t, y in enumerate(observations):
            self.predict()
            x, P, K = self.update(y)
            filtered_states[t] = x
            filtered_variances[t] = P
            kalman_gains[t] = K

        return filtered_states, filtered_variances, kalman_gains


class LocalLinearTrendKalman:
    """
    Kalman filter for local linear trend model.

    State: [level, trend]
    Observation: y_t = level_t + noise
    Level: level_t = level_{t-1} + trend_{t-1} + level_noise
    Trend: trend_t = trend_{t-1} + trend_noise
    """

    def __init__(self, Q_level, Q_trend, R, level0=0, trend0=0, P0=1):
        self.Q_level = Q_level
        self.Q_trend = Q_trend
        self.R = R

        # State vector [level, trend]
        self.x = np.array([level0, trend0])

        # State covariance
        self.P = np.eye(2) * P0

        # Transition matrix
        self.F = np.array([[1, 1], [0, 1]])

        # Observation matrix
        self.H = np.array([1, 0])

        # State noise covariance
        self.Q = np.diag([Q_level, Q_trend])

    def filter(self, observations):
        n = len(observations)
        filtered_levels = np.zeros(n)
        filtered_trends = np.zeros(n)

        for t, y in enumerate(observations):
            # Predict
            x_pred = self.F @ self.x
            P_pred = self.F @ self.P @ self.F.T + self.Q

            # Update
            S = self.H @ P_pred @ self.H.T + self.R
            K = P_pred @ self.H.T / S
            innovation = y - self.H @ x_pred

            self.x = x_pred + K * innovation
            self.P = (np.eye(2) - np.outer(K, self.H)) @ P_pred

            filtered_levels[t] = self.x[0]
            filtered_trends[t] = self.x[1]

        return filtered_levels, filtered_trends


def generate_local_level_data(n=200, level_init=100, Q=1, R=10, seed=42):
    """Generate local level model data."""
    np.random.seed(seed)

    true_level = np.zeros(n)
    observations = np.zeros(n)

    true_level[0] = level_init
    observations[0] = true_level[0] + np.random.randn() * np.sqrt(R)

    for t in range(1, n):
        true_level[t] = true_level[t-1] + np.random.randn() * np.sqrt(Q)
        observations[t] = true_level[t] + np.random.randn() * np.sqrt(R)

    return observations, true_level


def generate_local_trend_data(n=200, level_init=100, trend_init=0.5,
                              Q_level=0.5, Q_trend=0.01, R=5, seed=42):
    """Generate local linear trend model data."""
    np.random.seed(seed)

    true_level = np.zeros(n)
    true_trend = np.zeros(n)
    observations = np.zeros(n)

    true_level[0] = level_init
    true_trend[0] = trend_init
    observations[0] = true_level[0] + np.random.randn() * np.sqrt(R)

    for t in range(1, n):
        true_trend[t] = true_trend[t-1] + np.random.randn() * np.sqrt(Q_trend)
        true_level[t] = true_level[t-1] + true_trend[t-1] + np.random.randn() * np.sqrt(Q_level)
        observations[t] = true_level[t] + np.random.randn() * np.sqrt(R)

    return observations, true_level, true_trend


def demo_local_level():
    """Demonstrate local level Kalman filter."""
    print("\n" + "="*50)
    print("Local Level Model (Random Walk + Noise)")
    print("="*50)

    # Generate data
    Q_true, R_true = 1.0, 10.0
    observations, true_level = generate_local_level_data(Q=Q_true, R=R_true)

    # Filter with correct parameters
    kf = KalmanFilter(Q=Q_true, R=R_true, x0=observations[0], P0=R_true)
    filtered, variances, gains = kf.filter(observations)

    # Performance metrics
    mse_obs = np.mean((observations - true_level)**2)
    mse_filtered = np.mean((filtered - true_level)**2)

    print(f"\nTrue parameters: Q={Q_true}, R={R_true}")
    print(f"Signal-to-noise ratio: {Q_true/R_true:.4f}")
    print(f"\nFiltering performance:")
    print(f"  MSE (raw observations): {mse_obs:.4f}")
    print(f"  MSE (Kalman filtered):  {mse_filtered:.4f}")
    print(f"  Improvement: {(1 - mse_filtered/mse_obs)*100:.1f}%")
    print(f"\nSteady-state Kalman gain: {gains[-1]:.4f}")
    print(f"Theoretical steady-state: {(-R_true + np.sqrt(R_true**2 + 4*Q_true*R_true))/(2*R_true):.4f}")

    return observations, filtered, true_level, gains


def demo_local_trend():
    """Demonstrate local linear trend Kalman filter."""
    print("\n" + "="*50)
    print("Local Linear Trend Model")
    print("="*50)

    # Generate data
    observations, true_level, true_trend = generate_local_trend_data()

    # Filter
    kf = LocalLinearTrendKalman(
        Q_level=0.5, Q_trend=0.01, R=5,
        level0=observations[0], trend0=0, P0=10
    )
    filtered_level, filtered_trend = kf.filter(observations)

    # Performance
    level_corr = np.corrcoef(filtered_level, true_level)[0, 1]
    trend_corr = np.corrcoef(filtered_trend, true_trend)[0, 1]

    print(f"\nCorrelation with true components:")
    print(f"  Level: {level_corr:.4f}")
    print(f"  Trend: {trend_corr:.4f}")
    print(f"\nFinal estimates:")
    print(f"  Level: {filtered_level[-1]:.2f} (true: {true_level[-1]:.2f})")
    print(f"  Trend: {filtered_trend[-1]:.4f} (true: {true_trend[-1]:.4f})")

    return observations, filtered_level, filtered_trend, true_level, true_trend


def demo_parameter_sensitivity():
    """Demonstrate sensitivity to parameter misspecification."""
    print("\n" + "="*50)
    print("Parameter Sensitivity Analysis")
    print("="*50)

    # Generate data with Q=1, R=10
    observations, true_level = generate_local_level_data(Q=1.0, R=10.0)

    # Try different Q/R ratios
    print("\nMSE for different assumed Q/R ratios:")
    print(f"{'Q':<8} {'R':<8} {'Q/R':<10} {'MSE':<12} {'vs True':<12}")
    print("-"*50)

    for Q in [0.1, 0.5, 1.0, 2.0, 5.0]:
        for R in [5.0, 10.0, 20.0]:
            kf = KalmanFilter(Q=Q, R=R, x0=observations[0], P0=R)
            filtered, _, _ = kf.filter(observations)
            mse = np.mean((filtered - true_level)**2)
            print(f"{Q:<8.1f} {R:<8.1f} {Q/R:<10.4f} {mse:<12.4f} {'*Best*' if Q==1.0 and R==10.0 else ''}")


def plot_results(obs, filtered, true_level, gains, filename='.claude/kalman_demo_plot.png'):
    """Plot Kalman filter results."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Time series plot
    axes[0].plot(obs, 'b.', alpha=0.3, label='Observations', markersize=3)
    axes[0].plot(true_level, 'g-', label='True Level', linewidth=2)
    axes[0].plot(filtered, 'r-', label='Kalman Filtered', linewidth=2)
    axes[0].set_title('Local Level Kalman Filter')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Kalman gain convergence
    axes[1].plot(gains, 'b-')
    axes[1].axhline(y=gains[-1], color='r', linestyle='--', label=f'Steady state: {gains[-1]:.4f}')
    axes[1].set_title('Kalman Gain Convergence')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Kalman Gain')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def run_demo():
    """Run the complete Kalman filter demo."""
    print("="*60)
    print("Kalman Filter Demonstration")
    print("="*60)

    # Run demos
    obs, filtered, true_level, gains = demo_local_level()
    demo_local_trend()
    demo_parameter_sensitivity()

    # Plot
    try:
        fig = plot_results(obs, filtered, true_level, gains)
        fig.savefig('.claude/kalman_demo_plot.png', dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to .claude/kalman_demo_plot.png")
        plt.close(fig)
    except Exception as e:
        print(f"\nPlotting skipped: {e}")

    print("\n" + "="*60)
    print("Demo completed successfully!")
    print("="*60)


if __name__ == "__main__":
    run_demo()
