import numpy as np
import pandas as pd
from scipy.stats import norm

from riskbook.risk import (
    euler_risk_contributions,
    kupiec_and_christoffersen_tests,
    monte_carlo_var_cvar,
    parametric_var_cvar,
)


def test_euler_risk_contributions_sum_to_portfolio_vol():
    rng = np.random.default_rng(5)
    a = rng.normal(size=(6, 6))
    cov_mat = a @ a.T * 1e-4
    assets = [f"E{i}" for i in range(6)]
    cov = pd.DataFrame(cov_mat, index=assets, columns=assets)
    w = pd.Series([0.25, 0.15, 0.20, 0.10, 0.15, 0.15], index=assets)

    euler = euler_risk_contributions(cov, w)
    port_vol = float(np.sqrt(w.to_numpy() @ cov_mat @ w.to_numpy()))

    assert np.isclose(euler["component_risk"].sum(), port_vol, atol=1e-10)
    assert np.isclose(euler["pct_risk"].sum(), 1.0, atol=1e-10)


def test_parametric_and_monte_carlo_var_match_gaussian_theory():
    mu = np.array([0.0004, 0.0002])
    cov = np.array([[0.015**2, 0.0001], [0.0001, 0.02**2]])
    w = np.array([0.6, 0.4])

    var_p, cvar_p = parametric_var_cvar(mu, cov, w, alpha=0.05)
    var_mc, cvar_mc = monte_carlo_var_cvar(mu, cov, w, alpha=0.05, n_sims=60_000, seed=123)

    port_mu = float(w @ mu)
    port_vol = float(np.sqrt(w @ cov @ w))
    expected_var = -(port_mu + norm.ppf(0.05) * port_vol)
    assert np.isclose(var_p, expected_var, atol=1e-10)
    assert np.isclose(var_mc, var_p, rtol=0.03)
    assert np.isclose(cvar_mc, cvar_p, rtol=0.03)


def test_kupiec_and_christoffersen_detect_miscalibration_and_clustering():
    # Well-calibrated 5% breaches (10 out of 200, evenly spaced)
    good = np.zeros(200, dtype=int)
    good[::20] = 1
    res_good = kupiec_and_christoffersen_tests(good, alpha=0.05)
    assert res_good.kupiec_p_value > 0.50

    # Severely clustered breaches (10 consecutive out of 200: right count, terrible independence!)
    clustered = np.zeros(200, dtype=int)
    clustered[50:60] = 1
    res_clust = kupiec_and_christoffersen_tests(clustered, alpha=0.05)
    assert res_clust.kupiec_p_value > 0.50
    assert res_clust.christoffersen_ind_p_value < 1e-4

    # Excess breach rate (30 out of 200 = 15% vs 5% expected)
    bad = np.zeros(200, dtype=int)
    bad[::6] = 1
    res_bad = kupiec_and_christoffersen_tests(bad, alpha=0.05)
    assert res_bad.kupiec_p_value < 1e-4
