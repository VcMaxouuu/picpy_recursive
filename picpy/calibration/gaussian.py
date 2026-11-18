import numpy as np

from ..utils._array_api import get_namespace, standard_normal
from ._base import LambdaCalibrator


class GaussianCalibrator(LambdaCalibrator):
    """Regularization parameter for the Gaussian family (square-root loss).

    Null statistic: :math:`\\|X^\\top r\\|_\\infty / (\\sqrt{n}\\,\\|r\\|_2)`
    with :math:`r` the (centered when ``fit_intercept``) null response.
    """

    def _null_scores_mc_exact(self, X: np.ndarray, rng):
        n = X.shape[0]
        xp = get_namespace(X)
        for start, stop in self._iter_batches():
            r = standard_normal(rng, (n, stop - start), X)
            if self.fit_intercept:
                r = r - xp.mean(r, axis=0, keepdims=True)
            # Column norms written as sqrt(sum(r*r)), exactly what
            # np.linalg.norm(r, axis=0) computes, on any backend.
            norms = xp.sqrt(xp.sum(r * r, axis=0))
            yield start, stop, xp.abs(X.T @ r) / (norms * np.sqrt(n))
