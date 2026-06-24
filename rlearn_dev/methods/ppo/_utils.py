import numpy as np


def explained_variance(predictions, targets):
    targets_variance = np.var(targets)
    if targets_variance == 0:
        return np.nan
    return 1 - np.var(targets - predictions) / targets_variance


def recent_mean_reaches_threshold(values, threshold, window=3, min_count=4):
    if threshold is None or len(values) < min_count:
        return False
    return float(np.mean(values[-window:])) >= threshold
