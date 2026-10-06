import cvxpy as cp
import numpy as np
from scipy import sparse

from wddcontrol import contig, microsynth, sim
from wddcontrol.wdd import cumulative_wdd, wdd


def _connected(adj, nodes):
    sub = adj[nodes][:, nodes]
    return sparse.csgraph.connected_components(sub, directed=False)[0] == 1


def test_search_returns_connected_area_outside_buffer():
    w = sim.simulate(np.random.default_rng(11), nr=24, nc=24, scenario="local")
    X = w.counts[:, :w.n_pre]
    y = X[w.treated].sum(axis=0)
    g = contig.distance_graph(w.adj, w.xy)
    rr, cc = np.divmod(np.arange(w.nr * w.nc), w.nc)
    dist = np.hypot(rr - rr[w.treated].mean(), cc - cc[w.treated].mean())
    res = contig.search(X, y, g, ~w.excluded, top=5, time_limit=1, refine=2, refine_time=2, dist_to_treated=dist)
    for s in [res.best, res.scan, res.greedy] + res.ilp:
        assert len(s.nodes) > 0
        assert not w.excluded[s.nodes].any()
        assert _connected(w.adj, s.nodes)
    # the reported fit and violation match the selected units
    series = X[res.best.nodes].sum(axis=0)
    assert np.isclose(res.best.cost, res.criteria.fit(series))
    assert np.isclose(res.best.viol, res.criteria.violation(series))


def test_ordering_constraint_window_solution_is_connected():
    adj, xy = sim.grid_adjacency(10, 10)
    g = contig.distance_graph(adj, xy)
    rng = np.random.default_rng(3)
    X = rng.poisson(3.0, size=(100, 24)).astype(float)
    y = X[[0, 1, 10, 11]].sum(axis=0)
    w = next(contig.iter_windows(g, np.array([55]), X.sum(axis=1), 3 * y.sum(), radius=20.0))
    crit = contig.Criteria.from_series(y, dispersion=1.0)
    pos, status, gap = contig.solve_window(X[w.nodes], w.dist, w.closer, crit, True, time_limit=5,
                                           degree=np.diff(g.indptr)[w.nodes])
    assert pos[0] == 0  # the center is always selected
    assert _connected(adj, w.nodes[pos])


def test_criteria_thresholds():
    y = np.full(24, 10.0)
    c = contig.Criteria.from_series(y, dispersion=1.0, scales=(6,))
    assert np.isclose(c.tau, 24 * np.sqrt(2 * 20 / np.pi))
    assert np.isclose(c.level, np.sqrt(24 * 20))
    assert len(c.blocks) == 1 and c.blocks[0][0].shape == (4, 24)
    assert c.violation(y) == 0.0
    # dispersion scales every threshold by sqrt(phi)
    c2 = contig.Criteria.from_series(y, dispersion=4.0)
    assert np.isclose(c2.tau, 2 * c.tau) and np.isclose(c2.slope, 2 * c.slope)


def test_dispersion_estimate():
    rng = np.random.default_rng(0)
    assert contig.estimate_dispersion(rng.poisson(20, 200)) < 1.3
    lam = rng.gamma(2.0, 10.0, 200)
    assert contig.estimate_dispersion(rng.poisson(lam)) > 3


def test_rake_matches_convex_solution():
    rng = np.random.default_rng(3)
    X = rng.poisson(2.0, size=(300, 10)).astype(float)
    y = X[:15].sum(axis=0) * 1.1
    cal = microsynth.calibrate(X, y, 15)
    assert cal.exact and cal.intercept
    w = cp.Variable(300, nonneg=True)
    d = np.full(300, 15 / 300)
    cp.Problem(cp.Minimize(cp.sum(cp.rel_entr(w, d) - w + d)), [X.T @ w == y, cp.sum(w) == 15]).solve(solver=cp.CLARABEL)
    assert np.abs(cal.w - w.value).max() < 1e-3
    assert np.allclose(X.T @ cal.w, y, atol=1e-6)


def test_cumulative_wdd_ends_at_wdd():
    t_pre, c_pre = np.array([10, 12, 11]), np.array([9, 11, 13])
    t_post, c_post = np.array([8, 7, 9]), np.array([10, 12, 11])
    r = cumulative_wdd(t_pre, c_pre, t_post, c_post)
    e = wdd(t_pre.sum(), t_post.sum(), c_pre.sum(), c_post.sum())
    assert np.isclose(r["est"][-1], e.est) and np.isclose(r["se"][-1], e.se)


def test_wdd_scaled_control():
    e = wdd(100, 80, 300, 300, k=3.0)
    assert np.isclose(e.est, -20.0)
    assert np.isclose(e.se, np.sqrt(180 + 600 / 9))
