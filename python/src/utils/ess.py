import numpy as np
from scipy.signal import correlate

MAX_LAG = 2000


def compute_ess(values: np.ndarray) -> float:
    """Computes the ESS of the 1D array `values` using the same algorithm as Tracer."""
    trace = np.asarray(values, dtype=float)
    if trace.ndim != 1:
        raise ValueError("values must be a 1D array")

    sample_count = len(trace)
    if sample_count == 0:
        return float("nan")

    max_lag = min(sample_count, MAX_LAG)
    lag_indexes = np.arange(max_lag)
    lag_counts = sample_count - lag_indexes
    total = np.sum(trace)
    mean = total / sample_count
    cumulative_sum = np.concatenate(([0.0], np.cumsum(trace)))

    square_lagged_sums = correlate(trace, trace, mode="full", method="auto")
    square_lagged_sums = square_lagged_sums[sample_count - 1 : sample_count - 1 + max_lag]
    sum1 = cumulative_sum[sample_count - lag_indexes]
    sum2 = total - cumulative_sum[lag_indexes]
    auto_correlation = (
        square_lagged_sums - (sum1 + sum2) * mean + mean * mean * lag_counts
    )
    auto_correlation /= lag_counts

    integral_of_ac_function_times2 = 0.0
    for lag_index in range(max_lag):
        if lag_index == 0:
            integral_of_ac_function_times2 = auto_correlation[0]
        elif lag_index % 2 == 0:
            if auto_correlation[lag_index - 1] + auto_correlation[lag_index] > 0:
                integral_of_ac_function_times2 += 2.0 * (
                    auto_correlation[lag_index - 1] + auto_correlation[lag_index]
                )
            else:
                break

    act = integral_of_ac_function_times2 / auto_correlation[0]
    return sample_count / act
