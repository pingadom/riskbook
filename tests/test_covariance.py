import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from riskbook.covariance import ewma_covariance, ledoit_wolf_covariance, sample_covariance


def test_ledoit_wolf_matches_sklearn():
    rng = np.random.default_rng(7)
    x = rng.normal(0.0, 0.015, size=(120, 10))
    df = pd.DataFrame(x, columns=[f"A{i}" for i in range(10)])

    cov_ours, shrink_ours = ledoit_wolf_covariance(df)
    lw_ref = LedoitWolf().fit(x)

    assert np.isclose(shrink_ours, lw_ref.shrinkage_, atol=1e-8)
    assert np.allclose(cov_ours.to_numpy(), lw_ref.covariance_, atol=1e-8)


def test_covariance_estimators_are_symmetric_psd():
    rng = np.random.default_rng(13)
    df = pd.DataFrame(rng.normal(0.0, 0.02, size=(60, 8)), columns=[f"S{i}" for i in range(8)])

    for cov in (sample_covariance(df), ledoit_wolf_covariance(df)[0], ewma_covariance(df, lam=0.94)):
        mat = cov.to_numpy()
        assert np.allclose(mat, mat.T, atol=1e-12)
        eigvals = np.linalg.eigvalsh(mat)
        assert np.all(eigvals > 0.0)
