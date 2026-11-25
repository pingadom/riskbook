import numpy as np
import pandas as pd

from riskbook.optimisers import cvar_optimise, hrp_weights, min_variance
from riskbook.risk import historical_var_cvar


def test_two_asset_min_variance_matches_closed_form():
    s1, s2, rho = 0.02, 0.03, 0.25
    cov_mat = np.array([[s1**2, rho * s1 * s2], [rho * s1 * s2, s2**2]])
    cov = pd.DataFrame(cov_mat, index=["A", "B"], columns=["A", "B"])

    w = min_variance(cov, max_weight=1.0)
    # Closed-form 2-asset GMV weight for asset A
    w_a_exact = (s2**2 - rho * s1 * s2) / (s1**2 + s2**2 - 2 * rho * s1 * s2)
    assert np.isclose(w["A"], w_a_exact, atol=1e-6)
    assert np.isclose(w.sum(), 1.0, atol=1e-9)


def test_hrp_diagonal_covariance_equals_inverse_variance():
    vols = np.array([0.01, 0.02, 0.03, 0.04])
    cov = pd.DataFrame(np.diag(vols**2), index=["A", "B", "C", "D"], columns=["A", "B", "C", "D"])
    w = hrp_weights(cov, max_weight=1.0)

    expected = (1.0 / (vols**2))
    expected /= expected.sum()
    assert np.allclose(w.to_numpy(), expected, atol=1e-6)


def test_constraints_and_cvar_optimality():
    rng = np.random.default_rng(19)
    assets = [f"T{i}" for i in range(12)]
    sectors = {t: f"Sec{i % 4}" for i, t in enumerate(assets)}
    # Make T0..T2 much riskier than others
    scales = np.array([0.04, 0.04, 0.04] + [0.01] * 9)
    ret = pd.DataFrame(rng.normal(0.0, scales, size=(300, 12)), columns=assets)
    cov = ret.cov()

    w_mv = min_variance(cov, max_weight=0.15, sectors=sectors, max_sector_weight=0.32)
    w_hrp = hrp_weights(cov, max_weight=0.15, sectors=sectors, max_sector_weight=0.32)
    w_cvar = cvar_optimise(ret, alpha=0.05, max_weight=0.15, sectors=sectors, max_sector_weight=0.32)

    for w in (w_mv, w_hrp, w_cvar):
        assert np.isclose(w.sum(), 1.0, atol=1e-7)
        assert (w >= -1e-9).all()
        assert (w <= 0.15 + 1e-6).all()
        sec_sums = w.groupby(lambda t: sectors[t]).sum()
        assert (sec_sums <= 0.32 + 1e-5).all()

    # In-sample CVaR of CVaR-optimal portfolio must be <= equal-weight CVaR
    ew_r = ret.mean(axis=1)
    opt_r = ret.to_numpy() @ w_cvar.to_numpy()
    _, cvar_ew = historical_var_cvar(ew_r, alpha=0.05)
    _, cvar_opt = historical_var_cvar(opt_r, alpha=0.05)
    assert cvar_opt <= cvar_ew + 1e-6
