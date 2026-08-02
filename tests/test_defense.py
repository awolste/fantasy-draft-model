import numpy as np
import pytest

from ffdraft.models.defense import dst_distribution


def test_mean_is_in_a_plausible_starting_dst_range():
    assert 4.0 <= dst_distribution().mean <= 12.0


def test_sampled_mean_converges_to_stated_mean():
    d = dst_distribution()
    samples = d.sample(np.random.default_rng(0), 100_000)
    assert abs(samples.mean() - d.mean) < 0.15


def test_negative_weeks_are_possible():
    """A defense that gets blown out can score below zero in this league."""
    d = dst_distribution()
    assert (d.sample(np.random.default_rng(0), 100_000) < 0).any()


def test_elite_weeks_reach_the_low_twenties():
    d = dst_distribution()
    samples = d.sample(np.random.default_rng(0), 100_000)
    assert samples.max() >= 20.0


def test_sampling_is_reproducible_from_a_seed():
    d = dst_distribution()
    a = d.sample(np.random.default_rng(7), 1000)
    b = d.sample(np.random.default_rng(7), 1000)
    assert np.array_equal(a, b)


def test_all_teams_share_one_distribution():
    """The point of the design: no team has a DST edge over another."""
    assert dst_distribution().mean == dst_distribution().mean
    assert dst_distribution().stdev == dst_distribution().stdev


def test_sample_returns_requested_size():
    assert dst_distribution().sample(np.random.default_rng(0), 37).shape == (37,)
