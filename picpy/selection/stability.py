"""Stability selection by repeated subsampling."""

import copy
import warnings
from contextlib import nullcontext
from numbers import Integral, Real
from typing import Optional

import numpy as np
from joblib import Parallel, delayed
from pandas import DataFrame
from tqdm_joblib import tqdm_joblib

from ..calibration import LambdaCalibrator, GaussianCalibrator, BinomialCalibrator
from ..exceptions import ConvergenceWarning
from ..families import Gaussian, Binomial
from ..models._base import _BasePicModel
from ..utils import check_Xy
from ..utils._array_api import get_namespace, is_torch, to_numpy

_CALIBRATORS = {Gaussian: GaussianCalibrator, Binomial: BinomialCalibrator}


class StabilitySelection:
    """Stability analysis of the support selected by a picpy model.

    The model is refitted on random subsamples drawn without replacement.
    Each subsample is re-standardized and lambda is recalibrated before fitting.

    Parameters
    ----------
    model : LinearRegression or LogisticRegression
        Picpy model, cloned for each resample. Its lambda_ is recalibrated.
    calibrator : LambdaCalibrator or None, default=None
        Calibrator used on each subsample. None uses analytical calibration.
    n_resamples : int, default=500
        Number of subsamples.
    sample_fraction : float, default=0.8
        Fraction of observations per subsample.
    stratify : bool, default=True
        Preserve class proportions for binary responses.
    random_state : int, Generator or None, default=42
        Random seed for subsampling.
    n_jobs : int, default=-1
        Number of parallel jobs (-1 uses all cores).
    progress : bool, default=True
        Display a tqdm progress bar, including with parallel execution.

    Attributes
    ----------
    supports_ : ndarray (n_resamples, n_features)
        Boolean matrix of selected supports.
    selection_frequencies_ : ndarray (n_features,)
        Selection frequency of each feature.
    n_selected_ : ndarray (n_resamples,)
        Number of selected features per resample.
    subsample_indices_ : list of ndarray
        Observation indices for each subsample.
    lambdas_ : ndarray (n_resamples,)
        Calibrated regularization parameters.
    converged_ : ndarray (n_resamples,)
        Solver convergence flags.
    stability_ : dict
        Jaccard, Nogueira and support-size statistics.
    """

    def __init__(
        self,
        model: _BasePicModel,
        calibrator: LambdaCalibrator | None = None,
        n_resamples: int = 500,
        sample_fraction: float = 0.8,
        stratify: bool = True,
        random_state: int | np.random.Generator | None = 42,
        n_jobs: int = -1,
        progress: bool = True,
    ):
        self.model = model
        self.calibrator = calibrator
        self.n_resamples = n_resamples
        self.sample_fraction = sample_fraction
        self.stratify = stratify
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.progress = progress

        self.supports_: Optional[np.ndarray] = None
        self.selection_frequencies_: Optional[np.ndarray] = None
        self.n_selected_: Optional[np.ndarray] = None
        self.subsample_indices_: Optional[list] = None
        self.lambdas_: Optional[np.ndarray] = None
        self.converged_: Optional[np.ndarray] = None
        self.stability_: Optional[dict] = None
        self.n_features_in_: Optional[int] = None
        self.feature_names_in_: Optional[np.ndarray] = None
        self.is_fitted_: bool = False

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(model={self.model!r}, "
            f"calibrator={self.calibrator!r}, n_resamples={self.n_resamples}, "
            f"sample_fraction={self.sample_fraction}, stratify={self.stratify}, "
            f"random_state={self.random_state!r}, n_jobs={self.n_jobs}, "
            f"progress={self.progress})"
        )

    def _validate_params(self) -> None:
        if not isinstance(self.model, _BasePicModel):
            raise TypeError(
                "model must be a picpy model (e.g. LinearRegression); "
                f"got {type(self.model).__name__}."
            )
        expected = _CALIBRATORS[type(self.model.family)]
        if self.calibrator is not None:
            if not isinstance(self.calibrator, expected):
                raise TypeError(
                    f"{type(self.model).__name__} requires a "
                    f"{expected.__name__}; got {type(self.calibrator).__name__}."
                )
            if self.calibrator.fit_intercept != self.model.fit_intercept:
                raise ValueError(
                    f"calibrator.fit_intercept={self.calibrator.fit_intercept} "
                    f"does not match model.fit_intercept={self.model.fit_intercept}."
                )
        if (
            not isinstance(self.n_resamples, Integral)
            or isinstance(self.n_resamples, (bool, np.bool_))
            or self.n_resamples < 2
        ):
            raise ValueError(
                f"n_resamples must be an integer >= 2; got {self.n_resamples!r}."
            )
        if (
            not isinstance(self.sample_fraction, Real)
            or not np.isfinite(self.sample_fraction)
            or not 0.0 < self.sample_fraction < 1.0
        ):
            raise ValueError(
                f"sample_fraction must be in (0, 1); got {self.sample_fraction!r}."
            )
        if not isinstance(self.progress, (bool, np.bool_)):
            raise TypeError(f"progress must be a boolean; got {self.progress!r}.")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StabilitySelection":
        """Fit the model on every subsample and record selected supports."""
        self._validate_params()

        feature_names = X.columns.to_numpy() if isinstance(X, DataFrame) else None
        X, y = check_Xy(
            X,
            y,
            y_kind=self.model.family.y_kind,
            device=getattr(self.model, "device", None),
            dtype=getattr(self.model, "dtype", None),
        )
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = feature_names

        stratify = self.stratify and self.model.family.y_kind == "binary"
        rng = np.random.default_rng(self.random_state)
        y_host = to_numpy(y)
        indices = [
            _draw_subsample(rng, y_host, self.sample_fraction, stratify)
            for _ in range(self.n_resamples)
        ]
        seeds = rng.integers(np.iinfo(np.int32).max, size=self.n_resamples)

        if self.calibrator is None:
            calibrator = _CALIBRATORS[type(self.model.family)](
                method="analytical", fit_intercept=self.model.fit_intercept
            )
        else:
            calibrator = self.calibrator

        params = {**self.model.get_params(), "verbosity": 0}
        context = (
            tqdm_joblib(desc="Stability Selection", total=self.n_resamples)
            if self.progress
            else nullcontext()
        )

        if is_torch(X):
            # Tensors (typically on a GPU) are not shipped to worker
            # processes: the device already parallelizes each fit, so the
            # resamples run one after the other in this process.
            import torch

            iterator = zip(indices, seeds)
            if self.progress:
                from tqdm.auto import tqdm

                iterator = tqdm(
                    iterator, total=self.n_resamples, desc="Stability Selection"
                )
            results = []
            for idx, seed in iterator:
                rows = torch.as_tensor(idx, device=X.device)
                results.append(
                    _fit_subsample(
                        type(self.model), params, calibrator, int(seed), X[rows], y[rows]
                    )
                )
        else:
            with context:
                results = Parallel(n_jobs=self.n_jobs)(
                    delayed(_fit_subsample)(
                        type(self.model), params, calibrator, int(seed), X[idx], y[idx]
                    )
                    for idx, seed in zip(indices, seeds)
                )

        self.supports_ = np.array([res[0] for res in results], dtype=bool)
        self.lambdas_ = np.array([res[1] for res in results])
        self.converged_ = np.array([res[2] for res in results])
        self.subsample_indices_ = indices
        self.n_selected_ = self.supports_.sum(axis=1)
        self.selection_frequencies_ = self.supports_.mean(axis=0)
        self.stability_ = _stability_measures(self.supports_)
        self.is_fitted_ = True

        n_failed = int(np.sum(~self.converged_))
        if n_failed:
            warnings.warn(
                f"{type(self).__name__}: the solver did not converge on "
                f"{n_failed}/{self.n_resamples} resamples (see converged_).",
                ConvergenceWarning,
                stacklevel=2,
            )
        return self

    def get_support(self, threshold: float = 0.9) -> np.ndarray:
        """Return indices of features selected in at least threshold fraction of resamples."""
        self._check_is_fitted()
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1]; got {threshold!r}.")
        return np.flatnonzero(self.selection_frequencies_ >= threshold)

    def _check_is_fitted(self) -> None:
        if not self.is_fitted_:
            raise ValueError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call 'fit' before using this estimator."
            )

    def plot_supports(
        self,
        sort: bool = True,
        selected_only: bool = True,
        figsize: tuple = (12, 7),
        cmap: str = "Greys",
        ax=None,
    ):
        """Plot the feature selection matrix across resamples.

        Parameters
        ----------
        sort : bool, default=True
            Sort features by decreasing selection frequency.
        selected_only : bool, default=True
            Display only features selected at least once.
        figsize : tuple, default=(12, 7)
            Figure size when ax is None.
        cmap : str, default="Greys"
            Matplotlib colormap.
        ax : matplotlib.axes.Axes or None
            Existing axes. If None, create a new figure.

        Returns
        -------
        ax : matplotlib.axes.Axes
            Axes containing the selection heatmap.
        """
        import matplotlib.pyplot as plt

        self._check_is_fitted()

        frequencies = self.selection_frequencies_
        indices = np.arange(self.n_features_in_)

        if selected_only:
            indices = indices[frequencies > 0]

        if sort:
            indices = indices[np.argsort(-frequencies[indices], kind="stable")]

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        if len(indices) == 0:
            ax.text(
                0.5,
                0.5,
                "No features selected",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            return ax

        ax.imshow(
            self.supports_[:, indices],
            aspect="auto",
            cmap=cmap,
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )

        ax.set_xlabel("Feature index")
        ax.set_ylabel("Resample")
        ax.set_title(
            f"Stability Selection | "
            f"{len(indices)} features | "
            f"Mean Jaccard = {self.stability_['mean_jaccard']:.3f}"
        )

        step = max(1, int(np.ceil(len(indices) / 40)))
        ticks = np.arange(0, len(indices), step)
        ax.set_xticks(ticks)
        ax.set_xticklabels(indices[ticks], rotation=90, fontsize=8)

        ax.figure.tight_layout()
        return ax


def _draw_subsample(
    rng: np.random.Generator, y: np.ndarray, fraction: float, stratify: bool
) -> np.ndarray:
    """Draw sorted subsample indices without replacement."""
    n = y.shape[0]
    if not stratify:
        size = max(1, int(round(fraction * n)))
        return np.sort(rng.choice(n, size=size, replace=False))

    parts = []
    for label in np.unique(y):
        members = np.flatnonzero(y == label)
        size = min(len(members), max(1, int(round(fraction * len(members)))))
        parts.append(rng.choice(members, size=size, replace=False))
    return np.sort(np.concatenate(parts))


def _fit_subsample(
    model_cls: type,
    params: dict,
    calibrator: LambdaCalibrator,
    seed: int,
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, float, bool]:
    """Standardize, calibrate and fit one subsample."""
    xp = get_namespace(X)
    std = xp.std(X, axis=0, correction=0)
    X = (X - xp.mean(X, axis=0)) / xp.where(std == 0, 1.0, std)

    calibrator = copy.deepcopy(calibrator)
    calibrator.random_state = seed

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.filterwarnings(
            "ignore", message=".*does not appear to be standardized"
        )
        lambda_ = calibrator.compute(X)
        model = model_cls(**{**params, "lambda_": lambda_})
        model.fit(X, y)

    return to_numpy(model.coef_) != 0, lambda_, bool(model.converged_)


def _stability_measures(supports: np.ndarray) -> dict:
    """Compute pairwise Jaccard, Nogueira index and support-size distribution."""
    n_resamples, n_features = supports.shape
    S = supports.astype(float)
    sizes = S.sum(axis=1)

    inter = S @ S.T
    union = sizes[:, None] + sizes[None, :] - inter
    jaccard = np.divide(inter, union, out=np.ones_like(inter), where=union > 0)
    pairs = jaccard[np.triu_indices(n_resamples, k=1)]

    freq = S.mean(axis=0)
    k_bar = sizes.mean()
    denom = (k_bar / n_features) * (1.0 - k_bar / n_features)

    if denom > 0:
        var = n_resamples / (n_resamples - 1) * freq * (1.0 - freq)
        nogueira = float(1.0 - var.mean() / denom)
    else:
        nogueira = float("nan")

    values, counts = np.unique(sizes.astype(int), return_counts=True)
    return {
        "mean_jaccard": float(pairs.mean()),
        "std_jaccard": float(pairs.std()),
        "nogueira": nogueira,
        "mean_n_selected": float(sizes.mean()),
        "std_n_selected": float(sizes.std()),
        "n_selected_distribution": dict(zip(values.tolist(), counts.tolist())),
    }
