"""
Change-Point Detection Demo
===========================
Demonstrates various change-point detection methods.

Related notes:
- docs/en/change-detection/change-point.md
"""

import numpy as np
import matplotlib.pyplot as plt


def generate_data_with_changepoints(n=300, changepoints=[100, 200], seed=42):
    """Generate data with mean shifts at specified changepoints."""
    np.random.seed(seed)

    # Define segments
    segments = np.split(np.arange(n), changepoints)
    means = [10, 25, 15]  # Different mean for each segment
    stds = [2, 3, 2]

    data = np.zeros(n)
    true_means = np.zeros(n)

    for seg, mean, std in zip(segments, means, stds):
        data[seg] = mean + np.random.randn(len(seg)) * std
        true_means[seg] = mean

    return data, true_means, changepoints


def cusum(data, threshold=5.0):
    """
    CUSUM (Cumulative Sum) algorithm for change-point detection.

    Parameters:
    -----------
    data : array-like - Input time series
    threshold : float - Detection threshold

    Returns:
    --------
    changepoints : list - Detected changepoint indices
    cusum_pos : array - Positive CUSUM values
    cusum_neg : array - Negative CUSUM values
    """
    n = len(data)
    mean_est = np.mean(data[:min(50, n)])  # Use early data to estimate baseline

    cusum_pos = np.zeros(n)
    cusum_neg = np.zeros(n)
    changepoints = []

    for t in range(1, n):
        cusum_pos[t] = max(0, cusum_pos[t-1] + data[t] - mean_est - 0.5)
        cusum_neg[t] = max(0, cusum_neg[t-1] - data[t] + mean_est - 0.5)

        if cusum_pos[t] > threshold or cusum_neg[t] > threshold:
            changepoints.append(t)
            cusum_pos[t] = 0
            cusum_neg[t] = 0
            mean_est = data[t]  # Reset baseline

    return changepoints, cusum_pos, cusum_neg


def binary_segmentation(data, min_segment_length=10, threshold=None):
    """
    Binary segmentation algorithm for offline change-point detection.

    Recursively splits series at the point of maximum difference.
    """
    if threshold is None:
        threshold = 3 * np.std(data)

    n = len(data)

    def find_best_split(start, end):
        if end - start < 2 * min_segment_length:
            return None, 0

        best_pos = None
        best_stat = 0

        for pos in range(start + min_segment_length, end - min_segment_length):
            left_mean = np.mean(data[start:pos])
            right_mean = np.mean(data[pos:end])
            left_var = np.var(data[start:pos]) + 1e-10
            right_var = np.var(data[pos:end]) + 1e-10

            # Likelihood ratio statistic (simplified)
            stat = abs(right_mean - left_mean) * np.sqrt(
                (pos - start) * (end - pos) / (end - start)
            )

            if stat > best_stat:
                best_stat = stat
                best_pos = pos

        return best_pos, best_stat

    def recursive_segment(start, end, changepoints):
        pos, stat = find_best_split(start, end)

        if pos is not None and stat > threshold:
            changepoints.append(pos)
            recursive_segment(start, pos, changepoints)
            recursive_segment(pos, end, changepoints)

    changepoints = []
    recursive_segment(0, n, changepoints)
    return sorted(changepoints)


def pelt(data, penalty=None):
    """
    PELT (Pruned Exact Linear Time) algorithm.

    This is a simplified version focusing on mean change detection.
    For production use, consider the ruptures library.
    """
    n = len(data)

    if penalty is None:
        penalty = np.log(n) * np.var(data)

    # Cost function: negative log-likelihood for segment
    def segment_cost(start, end):
        if end <= start:
            return float('inf')
        segment = data[start:end]
        var = np.var(segment) + 1e-10
        return len(segment) * (np.log(var) + 1)

    # Dynamic programming
    F = np.zeros(n + 1)
    last_change = np.zeros(n + 1, dtype=int)

    for t in range(1, n + 1):
        candidates = []
        for s in range(t):
            cost = F[s] + segment_cost(s, t) + penalty
            candidates.append((cost, s))

        best = min(candidates, key=lambda x: x[0])
        F[t] = best[0]
        last_change[t] = best[1]

    # Backtrack to find changepoints
    changepoints = []
    pos = n
    while pos > 0:
        cp = last_change[pos]
        if cp > 0:
            changepoints.append(cp)
        pos = cp

    return sorted(changepoints)


def evaluate_detection(detected, true_changepoints, tolerance=10):
    """Evaluate changepoint detection performance."""
    true_set = set(true_changepoints)
    detected_set = set(detected)

    # True positives (within tolerance)
    tp = 0
    for d in detected:
        for t in true_set:
            if abs(d - t) <= tolerance:
                tp += 1
                break

    # False positives
    fp = len(detected) - tp

    # False negatives
    fn = len(true_changepoints) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def demo_cusum():
    """Demonstrate CUSUM algorithm."""
    print("\n" + "="*50)
    print("CUSUM (Online Detection)")
    print("="*50)

    data, true_means, true_cps = generate_data_with_changepoints()

    # Run CUSUM
    detected, cusum_pos, cusum_neg = cusum(data, threshold=10)

    print(f"\nTrue changepoints: {true_cps}")
    print(f"Detected changepoints: {detected}")

    # Evaluate
    metrics = evaluate_detection(detected, true_cps)
    print(f"\nPerformance:")
    print(f"  Precision: {metrics['precision']:.2f}")
    print(f"  Recall:    {metrics['recall']:.2f}")
    print(f"  F1 Score:  {metrics['f1']:.2f}")

    return data, detected, cusum_pos, cusum_neg


def demo_binary_segmentation():
    """Demonstrate binary segmentation."""
    print("\n" + "="*50)
    print("Binary Segmentation (Offline Detection)")
    print("="*50)

    data, true_means, true_cps = generate_data_with_changepoints()

    # Run binary segmentation
    detected = binary_segmentation(data, threshold=5)

    print(f"\nTrue changepoints: {true_cps}")
    print(f"Detected changepoints: {detected}")

    metrics = evaluate_detection(detected, true_cps)
    print(f"\nPerformance:")
    print(f"  Precision: {metrics['precision']:.2f}")
    print(f"  Recall:    {metrics['recall']:.2f}")
    print(f"  F1 Score:  {metrics['f1']:.2f}")

    return detected


def demo_pelt():
    """Demonstrate PELT algorithm."""
    print("\n" + "="*50)
    print("PELT Algorithm (Optimal Detection)")
    print("="*50)

    data, true_means, true_cps = generate_data_with_changepoints()

    # Run PELT with different penalties
    print("\nPenalty sensitivity:")
    for penalty_mult in [0.5, 1.0, 2.0, 5.0]:
        penalty = penalty_mult * np.log(len(data)) * np.var(data)
        detected = pelt(data, penalty=penalty)
        metrics = evaluate_detection(detected, true_cps)
        print(f"  Penalty={penalty_mult:.1f}x: {len(detected)} changepoints, F1={metrics['f1']:.2f}")

    # Default
    detected = pelt(data)
    print(f"\nDefault result: {detected}")

    return detected


def plot_results(data, true_cps, detected_cps, cusum_pos=None, cusum_neg=None,
                filename='.claude/changepoint_demo_plot.png'):
    """Plot changepoint detection results."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Data with changepoints
    axes[0].plot(data, 'b-', alpha=0.7, label='Data')
    for cp in true_cps:
        axes[0].axvline(x=cp, color='g', linestyle='--', alpha=0.7, label='True CP' if cp == true_cps[0] else '')
    for cp in detected_cps:
        axes[0].axvline(x=cp, color='r', linestyle='-', alpha=0.5, label='Detected CP' if cp == detected_cps[0] else '')
    axes[0].set_title('Changepoint Detection')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # CUSUM statistics
    if cusum_pos is not None:
        axes[1].plot(cusum_pos, 'b-', label='CUSUM+')
        axes[1].plot(cusum_neg, 'r-', label='CUSUM-')
        axes[1].axhline(y=10, color='k', linestyle='--', alpha=0.5, label='Threshold')
        axes[1].set_title('CUSUM Statistics')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def run_demo():
    """Run the complete change-point detection demo."""
    print("="*60)
    print("Change-Point Detection Demonstration")
    print("="*60)

    # Run demos
    data, cusum_detected, cusum_pos, cusum_neg = demo_cusum()
    binary_detected = demo_binary_segmentation()
    pelt_detected = demo_pelt()

    # Summary comparison
    true_cps = [100, 200]
    print("\n" + "="*50)
    print("Summary Comparison")
    print("="*50)
    print(f"{'Method':<25} {'Detected':<20} {'F1':<10}")
    print("-"*55)

    for method, detected in [('CUSUM', cusum_detected),
                             ('Binary Segmentation', binary_detected),
                             ('PELT', pelt_detected)]:
        metrics = evaluate_detection(detected, true_cps)
        print(f"{method:<25} {str(detected):<20} {metrics['f1']:<10.2f}")

    # Plot
    try:
        fig = plot_results(data, true_cps, cusum_detected, cusum_pos, cusum_neg)
        fig.savefig('.claude/changepoint_demo_plot.png', dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to .claude/changepoint_demo_plot.png")
        plt.close(fig)
    except Exception as e:
        print(f"\nPlotting skipped: {e}")

    print("\n" + "="*60)
    print("Demo completed successfully!")
    print("="*60)


if __name__ == "__main__":
    run_demo()
