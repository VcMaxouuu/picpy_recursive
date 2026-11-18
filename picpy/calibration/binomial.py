import numpy as np
from array_api_compat import device as _device

from ..utils._array_api import get_namespace, rademacher01
from ._base import LambdaCalibrator


class BinomialCalibrator(LambdaCalibrator):
    """Regularization parameter for the Binomial family.

    Null statistic: :math:`\\|X^\\top (y - \\hat p)\\|_\\infty /
    (n \\sqrt{\\hat p (1 - \\hat p)})` with :math:`y_i \\sim \\mathrm{Bernoulli}(1/2)`,
    and :math:`\\hat p = \\bar y` when ``fit_intercept`` (else :math:`1/2`).
    """

    def _null_scores_mc_exact(self, X: np.ndarray, rng):
        n = X.shape[0]
        xp = get_namespace(X)
        for start, stop in self._iter_batches():
            b = stop - start
            y_null = rademacher01(rng, (n, b), X)
            if self.fit_intercept:
                p_hat = xp.mean(xp.astype(y_null, X.dtype), axis=0)
            else:
                p_hat = xp.full((b,), 0.5, dtype=X.dtype, device=_device(X))
            num = xp.abs(X.T @ (y_null - p_hat))
            denom = xp.sqrt(p_hat * (1 - p_hat))
            # Degenerate draws (all 0 or all 1) get a zero score.
            safe = xp.where(denom > 0, denom, 1.0)
            scores = xp.where(denom > 0, num / (safe * n), 0.0)
            yield start, stop, scores
