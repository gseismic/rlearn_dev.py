import math

import numpy as np

from rlearn_dev.methods.ppo._utils import explained_variance, recent_mean_reaches_threshold


def test_explained_variance_is_one_for_perfect_prediction():
    targets = np.array([1.0, 2.0, 3.0])
    predictions = targets.copy()

    assert explained_variance(predictions, targets) == 1.0


def test_explained_variance_is_nan_for_constant_targets():
    targets = np.array([1.0, 1.0, 1.0])
    predictions = np.array([0.5, 1.0, 1.5])

    assert math.isnan(explained_variance(predictions, targets))


def test_recent_mean_reaches_threshold_requires_minimum_count():
    values = [0.9, 0.9, 0.9]

    assert recent_mean_reaches_threshold(values, 0.5) is False


def test_recent_mean_reaches_threshold_uses_recent_values_and_threshold():
    values = [0.0, 0.5, 0.7, 0.9]

    assert recent_mean_reaches_threshold(values, 0.7) is True
    assert recent_mean_reaches_threshold(values, 0.71) is False
