import numpy as np
from array_api_compat import device as _device

from ..utils import check_X, check_standardized
from ..utils._array_api import (
    get_namespace,
    make_rng,
    quantile,
    rademacher01,
    standard_normal,
    take_columns,
    to_numpy,
)

_METHODS = ("mc_exact", "mc_gaussian", "analytical")


class LambdaCalibrator:
    """
    Base class for the calibration of the regularization parameter.

    The regularization parameter is the ``(1 - alpha)``-quantile of the
    sup-norm of the loss gradient at :math:`\\beta = 0` under the null model.
    It must be computed on the standardized design matrix that is then used
    to fit the model.

    Parameters
    ----------
    alpha : float, default=0.05
        Nominal level.  The regularization parameter is set to the
        ``(1 - alpha)``-quantile of the null gradient norm distribution,
        controlling the family-wise type-I error rate.
    method : {"mc_exact", "mc_gaussian", "analytical"}, default="mc_exact"
        Estimation method:

        - ``"mc_exact"`` — Monte Carlo on the exact null gradient of the
          family's loss.
        - ``"mc_gaussian"`` — Monte Carlo on its Gaussian approximation
          :math:`X^\\top W / n`, :math:`W \\sim N(0, I_n)`.
        - ``"analytical"`` — Bonferroni closed-form approximation.
    n_simu : int, default=2000
        Number of Monte Carlo draws for ``"mc_exact"`` and ``"mc_gaussian"``.
        Ignored for ``"analytical"``.
    batch_size : int or None, default=None
        Number of Monte Carlo simulations processed at once by the
        ``"mc_exact"`` and ``"mc_gaussian"`` methods.  Because both methods
        allocate intermediate arrays of shape ``(n_samples, n_simu)``, memory
        scales as ``n * n_simu`` and can become prohibitive for very large
        *n*. ``None`` processes all draws in a single pass (lowest
        overhead, highest memory).  Ignored for ``"analytical"``.
    fit_intercept : bool, default=True
        Whether the model the parameter is calibrated for fits an intercept.
        Must match the model's ``fit_intercept``.  Only affects
        ``"mc_exact"``.
    random_state : int, numpy.random.Generator or None, default=None
        Seed or generator for the Monte Carlo draws.  With an int, every
        call to :meth:`compute` on the same *X* returns the same value.
    rng_backend : {"numpy", "device"}, default="numpy"
        Where the Monte Carlo draws are generated when *X* is a
        ``torch.Tensor``.  ``"numpy"`` draws on the CPU with NumPy and moves
        them to the tensor's device, so a seed gives the same lambda on
        NumPy, CPU tensors and GPU tensors.  ``"device"`` draws directly on
        the GPU with ``torch.Generator``: much faster for very large
        ``n_samples * n_simu``, but the draws (hence lambda) differ from the
        NumPy ones while following the same distribution.  Ignored for
        NumPy inputs.

    Notes
    -----
    *X* may be a NumPy array or a ``torch.Tensor`` on any device; the
    simulation then runs with PyTorch on that device.  Combine a GPU tensor
    with ``batch_size`` when ``n_samples * n_simu`` does not fit in GPU
    memory.

    Attributes
    ----------
    value : float or None
        The regularization parameter from the most recent call to
        :meth:`compute`.
    statistics : ndarray of shape (n_simu,) or None
        Simulated statistics from the most recent call to :meth:`compute`.
        For ``"mc_exact"`` these are null gradient norms; for
        ``"mc_gaussian"`` they are the Gaussian inf-norm draws.
        None when ``method="analytical"``.

    Notes
    -----
    Every Monte Carlo statistic is a maximum over columns of a per-column
    null score whose normalization depends only on the simulated response.
    :meth:`compute_column_scores` returns these per-column scores, so the
    parameter for any subset of columns follows from
    :meth:`lambda_from_scores` without new draws.  Standardization is
    column-wise, so restricting a standardized matrix to a subset of its
    columns gives the same columns as standardizing that subset.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        method: str = "mc_exact",
        n_simu: int = 2000,
        batch_size: int | None = None,
        fit_intercept: bool = True,
        random_state: int | np.random.Generator | None = None,
        rng_backend: str = "numpy",
    ):
        if method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS!r}; got {method!r}.")
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
        if not isinstance(n_simu, (int, np.integer)) or n_simu < 1:
            raise ValueError(f"n_simu must be a positive integer; got {n_simu!r}.")
        if batch_size is not None and batch_size < 1:
            raise ValueError(
                f"batch_size must be a positive integer or None; got {batch_size!r}."
            )
        self.alpha = alpha
        self.method = method
        self.n_simu = n_simu
        self.batch_size = batch_size
        self.fit_intercept = fit_intercept
        if rng_backend not in ("numpy", "device"):
            raise ValueError(
                f"rng_backend must be 'numpy' or 'device'; got {rng_backend!r}."
            )
        self.random_state = random_state
        self.rng_backend = rng_backend

        self.value: float | None = None
        self.statistics: np.ndarray | None = None

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(alpha={self.alpha}, method={self.method!r}, "
            f"n_simu={self.n_simu}, batch_size={self.batch_size!r}, "
            f"fit_intercept={self.fit_intercept}, "
            f"random_state={self.random_state!r}, "
            f"rng_backend={self.rng_backend!r})"
        )

    def _iter_batches(self):
        """Yield ``(start, stop)`` column ranges partitioning the ``n_simu`` draws.

        When ``batch_size`` is None the whole range is yielded as a single
        ``(0, n_simu)`` slice, reproducing the non-batched behaviour.
        """
        step = self.batch_size or self.n_simu
        for start in range(0, self.n_simu, step):
            yield start, min(start + step, self.n_simu)

    def compute(self, X: np.ndarray) -> float:
        """Compute the regularization parameter.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Standardized design matrix, the same one later passed to the
            model's ``fit``.  A warning is issued when the columns do not
            appear to be zero-mean, unit-variance standardized.

        Returns
        -------
        lambda_ : float
            Selected regularization parameter.
        """
        X = check_X(X)
        check_standardized(X, caller=type(self).__name__)

        if self.method == "analytical":
            self.value = self._compute_analytical(X)
            return self.value

        xp = get_namespace(X)
        rng = make_rng(self.random_state, X, self.rng_backend)
        stats = xp.empty(self.n_simu, dtype=X.dtype, device=_device(X))
        for start, stop, scores in self._null_score_batches(X, rng):
            stats[start:stop] = xp.max(scores, axis=0)
        self.value = quantile(stats, 1.0 - self.alpha)
        self.statistics = to_numpy(stats)
        return self.value

    def compute_column_scores(self, X: np.ndarray) -> np.ndarray:
        """Simulate the per-column null scores once.

        Uses the same draws as :meth:`compute` for a given ``random_state``,
        so ``lambda_from_scores(compute_column_scores(X))`` equals
        ``compute(X)``.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Standardized design matrix.

        Returns
        -------
        scores : ndarray of shape (n_simu, n_features)
            Null score of every column for every Monte Carlo draw.  Memory
            is ``8 * n_simu * n_features`` bytes.
        """
        if self.method == "analytical":
            raise ValueError(
                "compute_column_scores requires a Monte Carlo method; "
                "the analytical parameter only depends on the number of columns."
            )
        X = check_X(X)
        check_standardized(X, caller=type(self).__name__)

        xp = get_namespace(X)
        rng = make_rng(self.random_state, X, self.rng_backend)
        out = xp.empty((self.n_simu, X.shape[1]), dtype=X.dtype, device=_device(X))
        for start, stop, scores in self._null_score_batches(X, rng):
            out[start:stop] = scores.T
        return out

    def lambda_from_scores(
        self, scores: np.ndarray, columns: np.ndarray | None = None
    ) -> float:
        """Regularization parameter for a subset of columns from stored scores.

        Parameters
        ----------
        scores : ndarray of shape (n_simu, n_features)
            Output of :meth:`compute_column_scores`.
        columns : array-like of int or None, default=None
            Columns over which the null maximum is taken.  None uses all.

        Returns
        -------
        lambda_ : float
            ``(1 - alpha)``-quantile of the maximum over ``columns``.
        """
        sub = scores if columns is None else take_columns(scores, columns)
        if sub.shape[1] == 0:
            raise ValueError("columns must contain at least one column.")
        stats = get_namespace(sub).max(sub, axis=1)
        self.value = quantile(stats, 1.0 - self.alpha)
        self.statistics = to_numpy(stats)
        return self.value

    def _null_score_batches(self, X: np.ndarray, rng: np.random.Generator):
        """Yield ``(start, stop, scores)`` with scores of shape (p, stop - start)."""
        if self.method == "mc_exact":
            yield from self._null_scores_mc_exact(X, rng)
        else:
            yield from self._null_scores_mc_gaussian(X, rng)

    def _null_scores_mc_exact(self, X: np.ndarray, rng: np.random.Generator):
        """Per-column scores of the family-specific null gradient."""
        raise NotImplementedError

    def _null_scores_mc_gaussian(self, X: np.ndarray, rng: np.random.Generator):
        """Per-column scores under the Gaussian approximation:
        .. math::

            \\nabla \\ell \\approx \\frac{1}{n} X^\\top W,
            \\qquad W \\sim N(0, I_n),
        """
        n = X.shape[0]
        xp = get_namespace(X)
        for start, stop in self._iter_batches():
            W = standard_normal(rng, (n, stop - start), X)
            yield start, stop, xp.abs(X.T @ W) / n

    def _compute_analytical(self, X: np.ndarray) -> float:
        """Bonferroni closed-form approximation.

        .. math::

            \\hat\\lambda =
            \\sqrt{\\frac{1}{n}}\\,
            \\Phi^{-1}\\!\\left(1 - \\frac{\\alpha}{2p}\\right)
        """
        from scipy.stats import norm

        n, p = X.shape
        self.statistics = None
        return float(norm.ppf(1.0 - self.alpha / (2.0 * p)) / np.sqrt(n))
