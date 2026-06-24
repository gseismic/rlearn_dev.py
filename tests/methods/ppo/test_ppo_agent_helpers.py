import math

import numpy as np

from rlearn_dev.methods.ppo.draft.agent import (
    _explained_variance as draft_explained_variance,
    _recent_mean_reaches_threshold as draft_recent_mean_reaches_threshold,
)
from rlearn_dev.methods.ppo.naive.ppo_agent import (
    _explained_variance as naive_explained_variance,
    _recent_mean_reaches_threshold as naive_recent_mean_reaches_threshold,
)


def test_explained_variance_is_one_for_perfect_prediction_in_each_version():
    targets = np.array([1.0, 2.0, 3.0])
    predictions = targets.copy()

    assert naive_explained_variance(predictions, targets) == 1.0
    assert draft_explained_variance(predictions, targets) == 1.0


def test_explained_variance_is_nan_for_constant_targets_in_each_version():
    targets = np.array([1.0, 1.0, 1.0])
    predictions = np.array([0.5, 1.0, 1.5])

    assert math.isnan(naive_explained_variance(predictions, targets))
    assert math.isnan(draft_explained_variance(predictions, targets))


def test_recent_mean_reaches_threshold_requires_minimum_count_in_each_version():
    values = [0.9, 0.9, 0.9]

    assert naive_recent_mean_reaches_threshold(values, 0.5) is False
    assert draft_recent_mean_reaches_threshold(values, 0.5) is False


def test_recent_mean_reaches_threshold_uses_recent_values_in_each_version():
    values = [0.0, 0.5, 0.7, 0.9]

    assert naive_recent_mean_reaches_threshold(values, 0.7) is True
    assert naive_recent_mean_reaches_threshold(values, 0.71) is False
    assert draft_recent_mean_reaches_threshold(values, 0.7) is True
    assert draft_recent_mean_reaches_threshold(values, 0.71) is False
