"""Weighted displacement difference test (Wheeler and Ratcliffe 2018).

The estimate is a difference in differences in counts,

    WDD = (T1 - T0) - (C1 - C0) / k,

where T0, T1 are treated area counts before and after the intervention and
C0, C1 are control area counts. ``k`` scales a control area that holds k times
the treated area's crime (k = 1 for an equal-sized control). Under Poisson
counts the variance is T0 + T1 + (C0 + C1) / k^2.

For a weighted control (microsynth), C0 and C1 are weighted sums and the
Poisson variance of the control part is sum_i w_i^2 (c0_i + c1_i).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class WDD:
    est: float
    se: float
    low: float
    high: float
    z: float
    pct: float  # estimate as a share of the expected treated post count (T1 - est)


def wdd(t0: float, t1: float, c0: float, c1: float, k: float = 1.0, ctrl_var: float | None = None,
        alpha: float = 0.05) -> WDD:
    """WDD estimate with a normal-approximation confidence interval.

    ``ctrl_var`` overrides the control's variance (default (c0 + c1) / k^2).
    """
    est = (t1 - t0) - (c1 - c0) / k
    cv = (c0 + c1) / k**2 if ctrl_var is None else ctrl_var
    se = float(np.sqrt(t0 + t1 + cv))
    q = norm.ppf(1 - alpha / 2)
    counter = t1 - est
    pct = est / counter if counter > 0 else np.nan
    return WDD(est=est, se=se, low=est - q * se, high=est + q * se, z=est / se if se > 0 else np.nan, pct=pct)


def wdd_weighted(t0: float, t1: float, c0: np.ndarray, c1: np.ndarray, w: np.ndarray, alpha: float = 0.05) -> WDD:
    """WDD against a weighted control (weights already put the control on the treated scale)."""
    return wdd(t0, t1, float(w @ c0), float(w @ c1), k=1.0, ctrl_var=float((w**2) @ (c0 + c1)), alpha=alpha)


def cumulative_wdd(t_pre: np.ndarray, c_pre: np.ndarray, t_post: np.ndarray, c_post: np.ndarray,
                   c_pre_var: np.ndarray | None = None, c_post_var: np.ndarray | None = None,
                   alpha: float = 0.05) -> dict:
    """Running WDD estimate after each post period (the CompStat counter).

    The baseline is the average treated minus control difference per period
    over the ``len(t_pre)`` baseline periods. After m post periods,

        est_m = sum_{j <= m} (T_j - C_j) - m * mean_pre(T - C),

    the crimes prevented so far relative to the baseline gap, with Poisson
    variance sum_{j <= m} (T_j + C_j) + (m / B)^2 * sum_pre (T + C). When m
    equals the baseline length B this is the WDD over equal pre and post
    windows. ``c_*_var`` give the control's per-period variance when it is a
    weighted control (default: its counts).
    """
    t_pre, c_pre, t_post, c_post = (np.asarray(a, dtype=float) for a in (t_pre, c_pre, t_post, c_post))
    c_pre_var = c_pre if c_pre_var is None else np.asarray(c_pre_var, dtype=float)
    c_post_var = c_post if c_post_var is None else np.asarray(c_post_var, dtype=float)
    B = len(t_pre)
    m = np.arange(1, len(t_post) + 1)
    est = np.cumsum(t_post - c_post) - m * (t_pre - c_pre).sum() / B
    var = np.cumsum(t_post + c_post_var) + (m / B) ** 2 * (t_pre + c_pre_var).sum()
    se = np.sqrt(var)
    q = norm.ppf(1 - alpha / 2)
    return {"m": m, "est": est, "se": se, "low": est - q * se, "high": est + q * se}
