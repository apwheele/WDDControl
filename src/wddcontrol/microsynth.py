"""Microsynth-style calibration weights (Robbins, Saunders and Kilmer 2017).

Microsynth (the R package) builds a synthetic control for a set of treated
micro areas by survey calibration: every untreated unit gets a weight, the
weights are as close as possible to equal base weights, and the weighted sums
of the untreated units exactly reproduce the treated area's totals on the
matching variables (pre-period outcomes, optionally covariates). An intercept
constraint makes the weights sum to the number of treated units. With the
raking distance this is the entropy problem

    minimize   sum_i w_i log(w_i / d_i) - w_i + d_i
    subject to sum_i w_i a_i = b,  w >= 0,

whose solution has the form w_i = d_i exp(a_i' lambda). ``rake`` finds
lambda by Newton's method on the dual, the raking algorithm of survey
calibration (Deville and Sarndal 1992). Like microsynth, ``calibrate`` falls
back when exact calibration is infeasible: first dropping the intercept
constraint, then (rarely) matching approximately with an L1 penalty, solved
with cvxpy.
"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np


@dataclass
class Calibration:
    w: np.ndarray
    exact: bool
    intercept: bool  # whether the weights sum to the number of treated units
    max_abs_err: float
    status: str


def rake(A: np.ndarray, b: np.ndarray, d: np.ndarray, tol: float = 1e-7, max_iter: int = 100) -> tuple[np.ndarray, bool]:
    """Raking weights w = d exp(A lambda) with A' w = b, by damped Newton steps on the dual.

    Constraints are scaled by their targets first. Returns (weights,
    converged); the iteration does not converge when calibration is
    infeasible.
    """
    s = np.where(np.abs(b) > 0, np.abs(b), 1.0)
    A, b = A / s, b / s
    lam = np.zeros(A.shape[1])

    def dual(lm):
        eta = np.clip(A @ lm, -700, 700)
        w = d * np.exp(eta)
        return w.sum() - b @ lm, w

    f, w = dual(lam)
    for _ in range(max_iter):
        g = A.T @ w - b
        if np.abs(g).max() < tol:
            return w, True
        H = A.T @ (A * w[:, None])
        step = np.linalg.solve(H + 1e-12 * np.trace(H) * np.eye(len(H)), g)
        t = 1.0
        while t > 1e-12:
            f_new, w_new = dual(lam - t * step)
            if f_new <= f - 1e-4 * t * (g @ step):
                break
            t /= 2
        else:
            return w, False
        lam, f, w = lam - t * step, f_new, w_new
    return w, np.abs(A.T @ w - b).max() < 1e-4


def calibrate(X: np.ndarray, y: np.ndarray, n_treated: float, covars: np.ndarray | None = None,
              covars_target: np.ndarray | None = None, penalty: float = 1e3) -> Calibration:
    """Raking weights for the untreated units in ``X`` (n_units, P) matching ``y`` (P,).

    Rows of ``X`` are untreated (donor) units. Base weights are
    ``n_treated / n_units``. Returns the weights and how exactly they match.
    """
    n = X.shape[0]
    A = [np.asarray(X, dtype=float)]
    b = [np.asarray(y, dtype=float)]
    if covars is not None:
        A.append(np.asarray(covars, dtype=float).reshape(n, -1))
        b.append(np.asarray(covars_target, dtype=float).ravel())
    A = np.hstack(A)
    b = np.concatenate(b)
    d = np.full(n, n_treated / n)
    for intercept in (True, False):
        Ai = np.hstack([np.ones((n, 1)), A]) if intercept else A
        bi = np.concatenate([[float(n_treated)], b]) if intercept else b
        w, ok = rake(Ai, bi, d)
        err = float(np.abs(Ai.T @ w - bi).max())
        if ok and err < 0.01:
            return Calibration(w=w, exact=True, intercept=intercept, max_abs_err=err, status="raking")
    # Approximate calibration: penalize the outcome misfit (relative to each target).
    s = np.where(np.abs(b) > 0, np.abs(b), 1.0)
    wv = cp.Variable(n, nonneg=True)
    dist = cp.sum(cp.rel_entr(wv, d) - wv + d)
    prob = cp.Problem(cp.Minimize(dist / n_treated + penalty * cp.norm1((A / s).T @ wv - b / s)))
    try:
        prob.solve(solver=cp.CLARABEL)
    except cp.SolverError:
        prob.solve(solver=cp.SCS, max_iters=20000)
    w = np.maximum(wv.value, 0)
    return Calibration(w=w, exact=False, intercept=False, max_abs_err=float(np.abs(A.T @ w - b).max()),
                       status=prob.status)
