"""Small-sample statistics for success/fail trials (binomial intervals, paired comparisons).

Everything here works on counts or 0/1 arrays, has no state and depends only on numpy/scipy, so
analyze_trials.py and the tests can use it directly.

    wilson_interval(k, n)            score interval for one success rate (the one we report)
    clopper_pearson_interval(k, n)   exact interval, for reference (conservative)
    mcnemar_exact(b, c)              exact and mid-p McNemar test on the discordant pair counts
    discordant_counts(base, other)   2x2 paired table from two aligned 0/1 outcome arrays
    paired_bootstrap_diff(x, y)      percentile bootstrap CI for mean(x) - mean(y) over pairs

References
    Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference.
        JASA 22(158), 209-212.
    Clopper, C. J., Pearson, E. S. (1934). The use of confidence or fiducial limits illustrated in
        the case of the binomial. Biometrika 26(4), 404-413.
    Brown, L. D., Cai, T. T., DasGupta, A. (2001). Interval estimation for a binomial proportion.
        Statistical Science 16(2), 101-133. (Recommends Wilson over Wald for small n.)
    McNemar, Q. (1947). Note on the sampling error of the difference between correlated
        proportions or percentages. Psychometrika 12(2), 153-157.
    Fagerland, M. W., Lydersen, S., Laake, P. (2013). The McNemar test for binary matched-pairs
        data: mid-p and asymptotic are better than exact conditional. BMC Medical Research
        Methodology 13, 91.
    Efron, B., Tibshirani, R. J. (1993). An Introduction to the Bootstrap. Chapman & Hall,
        ch. 13 (percentile intervals).
"""

from __future__ import annotations

import numbers
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats as _sps

_BOOT_CHUNK = 2000  # resamples drawn per batch, so memory stays bounded for long trial lists


def _is_count(value: Any) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, bool) and value >= 0


def _check_counts(k: int, n: int) -> None:
    if not (_is_count(k) and _is_count(n)):
        raise ValueError("k and n must be non-negative integers, got k=%r n=%r" % (k, n))
    if n <= 0:
        raise ValueError("n must be positive, got %d" % n)
    if not 0 <= k <= n:
        raise ValueError("k must be in [0, n], got k=%d n=%d" % (k, n))


def _alpha(confidence: float) -> float:
    """alpha = 1 - confidence, after checking the confidence level."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1), got %r" % confidence)
    return 1.0 - confidence


def wilson_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n (no continuity correction).

    Inverts the score test, so unlike the Wald interval it never leaves [0, 1] and keeps close
    to nominal coverage at the n of a few tens and rates near 0 or 1 that robot trials produce.
    """
    _check_counts(k, n)
    alpha = _alpha(confidence)
    z = float(_sps.norm.ppf(1.0 - alpha / 2.0))
    z2 = z * z
    denom = n + z2
    centre = (k + z2 / 2.0) / denom
    half = z * np.sqrt(k * (n - k) / n + z2 / 4.0) / denom
    # The bounds are exactly 0 and 1 at the extremes; the formula only gets there up to rounding.
    low = 0.0 if k == 0 else max(0.0, centre - half)
    high = 1.0 if k == n else min(1.0, centre + half)
    return float(low), float(high)


def clopper_pearson_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Clopper-Pearson 'exact' interval from beta quantiles; coverage >= nominal, often well above."""
    _check_counts(k, n)
    alpha = _alpha(confidence)
    low = 0.0 if k == 0 else float(_sps.beta.ppf(alpha / 2.0, k, n - k + 1))
    high = 1.0 if k == n else float(_sps.beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return low, high


@dataclass(frozen=True)
class McNemarResult:
    """b and c are the two discordant cells of the paired 2x2 table (see discordant_counts)."""

    b: int
    c: int
    p_exact: float  # two-sided exact conditional (binomial) p-value
    p_mid: float  # two-sided mid-p: the exact p minus the point probability of the observed table

    @property
    def n_discordant(self) -> int:
        return self.b + self.c


def mcnemar_exact(b: int, c: int) -> McNemarResult:
    """Exact McNemar test: under H0 each discordant pair is equally likely to go either way.

    Conditional on n = b + c discordant pairs, min(b, c) ~ Binomial(n, 1/2), so
        p_exact = min(1, 2 * P(X <= min(b, c)))
        p_mid   = 2 * P(X <= min(b, c)) - P(X = min(b, c))
    The exact test is conservative; the mid-p version keeps close to the nominal level and is
    what Fagerland et al. (2013) recommend. With no discordant pairs both are 1.
    """
    for name, value in (("b", b), ("c", c)):
        if not _is_count(value):
            raise ValueError("%s must be a non-negative integer, got %r" % (name, value))
    b, c = int(b), int(c)
    n = b + c
    if n == 0:
        return McNemarResult(b, c, 1.0, 1.0)
    low = min(b, c)
    tail = float(_sps.binom.cdf(low, n, 0.5))
    point = float(_sps.binom.pmf(low, n, 0.5))
    return McNemarResult(b, c, min(1.0, 2.0 * tail), min(1.0, 2.0 * tail - point))


@dataclass(frozen=True)
class PairedTable:
    """Paired outcomes of a method against a baseline on the same (pose, round) slots."""

    both_success: int
    b: int  # baseline succeeded, method failed
    c: int  # baseline failed, method succeeded
    both_fail: int

    @property
    def n_pairs(self) -> int:
        return self.both_success + self.b + self.c + self.both_fail


def _as_binary(values: Sequence[int] | np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError("%s must be one-dimensional" % name)
    if arr.size and not np.isin(arr, (0, 1)).all():
        raise ValueError("%s must contain only 0/1 (or False/True)" % name)
    return arr.astype(np.int64)


def discordant_counts(baseline: Sequence[int] | np.ndarray, method: Sequence[int] | np.ndarray) -> PairedTable:
    """2x2 paired table from aligned 0/1 outcomes: entry i of both arrays is the same slot."""
    base = _as_binary(baseline, "baseline")
    other = _as_binary(method, "method")
    if base.shape != other.shape:
        raise ValueError("baseline and method must have the same length, got %d and %d" % (base.size, other.size))
    return PairedTable(
        both_success=int(np.sum((base == 1) & (other == 1))),
        b=int(np.sum((base == 1) & (other == 0))),
        c=int(np.sum((base == 0) & (other == 1))),
        both_fail=int(np.sum((base == 0) & (other == 0))),
    )


@dataclass(frozen=True)
class PairedDiff:
    estimate: float  # mean(x) - mean(y) over the pairs
    low: float
    high: float
    n_pairs: int
    n_boot: int
    confidence: float


def paired_bootstrap_diff(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    confidence: float = 0.95,
    n_boot: int = 10000,
    seed: int = 0,
) -> PairedDiff:
    """Percentile bootstrap CI for mean(x) - mean(y), resampling whole pairs with replacement.

    Resampling the pair index keeps each slot's two outcomes together, so the interval reflects
    the paired design (what McNemar tests) rather than two independent samples. With 0/1 data
    and a few tens of pairs the bootstrap distribution is lumpy and the interval is approximate;
    report it beside the exact McNemar p, not instead of it.
    """
    alpha = _alpha(confidence)
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    if xs.ndim != 1 or xs.shape != ys.shape:
        raise ValueError("x and y must be one-dimensional and of equal length")
    n = xs.size
    if n == 0:
        raise ValueError("need at least one pair")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    d = xs - ys
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for start in range(0, n_boot, _BOOT_CHUNK):
        stop = min(start + _BOOT_CHUNK, n_boot)
        idx = rng.integers(0, n, size=(stop - start, n))
        boots[start:stop] = d[idx].mean(axis=1)
    low, high = np.quantile(boots, [alpha / 2.0, 1.0 - alpha / 2.0])
    return PairedDiff(float(d.mean()), float(low), float(high), int(n), int(n_boot), float(confidence))
