"""Known-value tests for tools/stats.py against closed forms, scipy and statsmodels."""

import math

import numpy as np
import pytest
from scipy import stats as sps
from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar
from statsmodels.stats.proportion import proportion_confint

import stats

COUNTS = [(0, 1), (1, 1), (0, 10), (1, 10), (5, 10), (10, 10), (7, 30), (24, 30), (29, 30), (50, 100), (3, 1000)]
CONFIDENCES = [0.90, 0.95, 0.99]


@pytest.mark.parametrize("confidence", CONFIDENCES)
@pytest.mark.parametrize("k,n", COUNTS)
def test_wilson_matches_statsmodels_and_scipy(k, n, confidence):
    low, high = stats.wilson_interval(k, n, confidence)
    sm_low, sm_high = proportion_confint(k, n, alpha=1.0 - confidence, method="wilson")
    sp = sps.binomtest(k, n).proportion_ci(confidence_level=confidence, method="wilson")
    assert low == pytest.approx(sm_low, abs=1e-12) and high == pytest.approx(sm_high, abs=1e-12)
    assert low == pytest.approx(sp.low, abs=1e-12) and high == pytest.approx(sp.high, abs=1e-12)
    assert 0.0 <= low <= k / n <= high <= 1.0


def test_wilson_closed_forms():
    z2 = sps.norm.ppf(0.975) ** 2
    assert stats.wilson_interval(0, 10) == (0.0, pytest.approx(z2 / (10 + z2), abs=1e-15))
    assert stats.wilson_interval(10, 10) == (pytest.approx(10 / (10 + z2), abs=1e-15), 1.0)
    # 24/30 at 95 %: the score formula evaluated with mpmath at 30 digits
    low, high = stats.wilson_interval(24, 30)
    assert low == pytest.approx(0.626943, abs=5e-6) and high == pytest.approx(0.904949, abs=5e-6)


@pytest.mark.parametrize("confidence", CONFIDENCES)
@pytest.mark.parametrize("k,n", COUNTS)
def test_clopper_pearson_matches_statsmodels_and_scipy(k, n, confidence):
    low, high = stats.clopper_pearson_interval(k, n, confidence)
    sm_low, sm_high = proportion_confint(k, n, alpha=1.0 - confidence, method="beta")
    sp = sps.binomtest(k, n).proportion_ci(confidence_level=confidence, method="exact")
    assert low == pytest.approx(sm_low, abs=1e-12) and high == pytest.approx(sm_high, abs=1e-12)
    assert low == pytest.approx(sp.low, abs=1e-12) and high == pytest.approx(sp.high, abs=1e-12)


def test_clopper_pearson_closed_forms_and_width():
    # k = 0 and k = n have closed forms: 1 - (alpha/2)^(1/n) and (alpha/2)^(1/n)
    assert stats.clopper_pearson_interval(0, 30) == (0.0, pytest.approx(1 - 0.025 ** (1 / 30), abs=1e-14))
    assert stats.clopper_pearson_interval(30, 30) == (pytest.approx(0.025 ** (1 / 30), abs=1e-14), 1.0)
    for k, n in COUNTS:  # the exact interval is the conservative one
        w_low, w_high = stats.wilson_interval(k, n)
        cp_low, cp_high = stats.clopper_pearson_interval(k, n)
        assert cp_high - cp_low >= w_high - w_low - 1e-12


@pytest.mark.parametrize("func", [stats.wilson_interval, stats.clopper_pearson_interval])
@pytest.mark.parametrize("args", [(1, 0), (-1, 5), (6, 5), (1.5, 5), (2.0, 5), (True, 5), (1, 5, 1.0), (1, 5, 0.0)])
def test_intervals_reject_bad_input(func, args):  # (k, n[, confidence])
    with pytest.raises(ValueError):
        func(*args)


@pytest.mark.parametrize(
    "b,c,p_exact,p_mid",
    [
        (0, 0, 1.0, 1.0),
        (2, 8, 112 / 1024, 67 / 1024),  # P(X<=2) = 56/1024, P(X=2) = 45/1024, n = 10
        (8, 2, 112 / 1024, 67 / 1024),  # symmetric in b and c
        (0, 15, 2 / 32768, 1 / 32768),
        (6, 3, 260 / 512, 176 / 512),  # P(X<=3) = 130/512, P(X=3) = 84/512, n = 9
        (2, 3, 1.0, 22 / 32),  # 2 P(X<=2) = 1 exactly, P(X=2) = 10/32
        (4, 4, 1.0, 1.0),
    ],
)
def test_mcnemar_known_values(b, c, p_exact, p_mid):
    result = stats.mcnemar_exact(b, c)
    assert (result.b, result.c, result.n_discordant) == (b, c, b + c)
    assert result.p_exact == pytest.approx(p_exact, rel=1e-12, abs=1e-15)
    assert result.p_mid == pytest.approx(p_mid, rel=1e-12, abs=1e-15)


def test_mcnemar_matches_statsmodels_and_binomtest():
    for b in range(13):
        for c in range(13):
            result = stats.mcnemar_exact(b, c)
            ref = sm_mcnemar([[5, b], [c, 5]], exact=True).pvalue
            assert result.p_exact == pytest.approx(ref, rel=1e-12, abs=1e-15), (b, c)
            if b + c:
                assert result.p_exact == pytest.approx(sps.binomtest(b, b + c, 0.5).pvalue, rel=1e-12)
                low = min(b, c)
                mid = 2 * (sps.binom.cdf(low - 1, b + c, 0.5) + 0.5 * sps.binom.pmf(low, b + c, 0.5))
                assert result.p_mid == pytest.approx(min(1.0, mid), rel=1e-12, abs=1e-15)
            assert 0.0 < result.p_mid <= result.p_exact <= 1.0


@pytest.mark.parametrize("b,c", [(-1, 2), (1.5, 2), (2.0, 1), (2, None), (True, 1)])
def test_mcnemar_rejects_bad_counts(b, c):
    with pytest.raises(ValueError):
        stats.mcnemar_exact(b, c)


def test_discordant_counts():
    baseline = [1, 1, 1, 0, 0, 1, 0, 1]
    method = [1, 0, 1, 1, 0, 0, 0, 1]
    table = stats.discordant_counts(baseline, method)
    assert (table.both_success, table.b, table.c, table.both_fail) == (3, 2, 1, 2)
    assert table.n_pairs == 8
    assert stats.discordant_counts(np.array([True, False]), np.array([False, False])).b == 1
    with pytest.raises(ValueError):
        stats.discordant_counts([1, 0], [1])
    with pytest.raises(ValueError):
        stats.discordant_counts([1, 2], [1, 0])


def test_paired_bootstrap_is_deterministic_and_centred():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 30)
    x = rng.integers(0, 2, 30)
    first = stats.paired_bootstrap_diff(x, y, n_boot=5000, seed=7)
    again = stats.paired_bootstrap_diff(x, y, n_boot=5000, seed=7)
    assert first == again
    assert first.estimate == pytest.approx(x.mean() - y.mean())
    assert first.low <= first.estimate <= first.high
    assert (first.n_pairs, first.n_boot, first.confidence) == (30, 5000, 0.95)


def test_paired_bootstrap_matches_scipy_percentile():
    # 30 pairs, method better on 6 slots, worse on 2: the CI must agree with scipy's paired
    # percentile bootstrap to within one discrete step of the statistic (1/30).
    y = np.array([1] * 20 + [0] * 10)
    x = np.array([1] * 18 + [0] * 2 + [1] * 6 + [0] * 4)
    ours = stats.paired_bootstrap_diff(x, y, n_boot=20000, seed=0)
    ref = sps.bootstrap(
        (x, y),
        lambda a, b, axis=-1: np.mean(a, axis=axis) - np.mean(b, axis=axis),
        paired=True,
        vectorized=True,
        n_resamples=20000,
        method="percentile",
        random_state=np.random.default_rng(0),
    ).confidence_interval
    assert ours.estimate == pytest.approx(4 / 30)
    assert ours.low == pytest.approx(ref.low, abs=1 / 30 + 1e-9)
    assert ours.high == pytest.approx(ref.high, abs=1 / 30 + 1e-9)


def test_paired_bootstrap_degenerate_and_invalid():
    same = stats.paired_bootstrap_diff([1, 0, 1], [1, 0, 1], n_boot=100)
    assert (same.estimate, same.low, same.high) == (0.0, 0.0, 0.0)
    shifted = stats.paired_bootstrap_diff([1, 1, 1, 1], [0, 0, 0, 0], n_boot=100)
    assert (shifted.low, shifted.high) == (1.0, 1.0)
    with pytest.raises(ValueError):
        stats.paired_bootstrap_diff([], [])
    with pytest.raises(ValueError):
        stats.paired_bootstrap_diff([1, 0], [1])
    with pytest.raises(ValueError):
        stats.paired_bootstrap_diff([1, 0], [1, 0], n_boot=0)


def test_paired_bootstrap_chunks_equal_one_draw():
    # 4500 resamples span three internal chunks; they must be the draws of one big batch
    x, y = np.arange(40) % 2, (np.arange(40) // 3) % 2
    result = stats.paired_bootstrap_diff(x, y, n_boot=4500, seed=3)
    idx = np.random.default_rng(3).integers(0, 40, size=(4500, 40))
    low, high = np.quantile((x - y)[idx].mean(axis=1), [0.025, 0.975])
    assert (result.low, result.high) == (low, high)
    assert math.isfinite(result.low) and result.low < result.high
