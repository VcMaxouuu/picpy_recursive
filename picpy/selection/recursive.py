"""Recursive discovery of disjoint predictive supports using PIC."""

import copy
import warnings
from numbers import Integral, Real
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn.linear_model import LinearRegression as SkLinearRegression
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, log_loss
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from ..calibration import LambdaCalibrator, GaussianCalibrator, BinomialCalibrator
from ..exceptions import ConvergenceWarning
from ..families import Gaussian, Binomial
from ..models._base import _BasePicModel
from ..utils import check_Xy
from ..utils._array_api import flatnonzero, is_torch, take_columns, to_numpy

_CALIBRATORS = {Gaussian: GaussianCalibrator, Binomial: BinomialCalibrator}


class RecursiveSelection:
    """Discover successive disjoint supports using PIC and unpenalized GLM refits.

    At each iteration, PIC selects a support among the remaining features.
    An unpenalized GLM is then fitted on the selected variables and evaluated
    on validation data. Selected features are removed before the next iteration.

    PIC always uses refit=False, regardless of the supplied model configuration.

    Parameters
    ----------
    model : LinearRegression or LogisticRegression
        Picpy model used exclusively for feature selection.
    calibrator : LambdaCalibrator or None, default=None
        Calibrator used at each iteration. None uses analytical calibration.
    max_iter : int, default=20
        Maximum number of recursive iterations.
    min_features : int, default=1
        Stop when fewer than this many features remain.
    min_selected : int, default=1
        Stop when PIC selects fewer than this many features.
    random_state : int or None, default=42
        Random seed for Monte Carlo calibration.
    reuse_null_draws : bool, default=True
        With a Monte Carlo calibrator, simulate the per-column null scores
        once on the full standardized design and obtain each iteration's
        parameter as the ``(1 - alpha)``-quantile of the maximum over the
        remaining columns.  This has the same null distribution as
        re-simulating at each iteration, costs a single simulation, and makes
        ``lambdas_`` non-increasing.  It needs ``n_simu * n_features`` floats
        of memory.  False re-simulates at every iteration.
    progress : bool, default=True
        Display a progress bar.

    Attributes
    ----------
    supports_ : list of ndarray
        Selected original feature indices at each iteration.
    n_selected_ : ndarray
        Number of selected features per iteration.
    lambdas_ : ndarray
        Calibrated regularization parameters.
    converged_ : ndarray
        PIC convergence flags.
    refit_models_ : list
        Unpenalized sklearn GLM fitted at each iteration.
    scalers_ : list
        StandardScaler fitted on each selected support.
    results_ : DataFrame
        Iteration-level results and predictive metrics.
    remaining_features_ : ndarray
        Features not selected after the final iteration.
    n_features_in_ : int
        Original number of features.
    feature_names_in_ : ndarray or None
        Feature names when X is a DataFrame.
    is_fitted_ : bool
        Whether recursive selection has been fitted.
    """

    def __init__(
        self,
        model: _BasePicModel,
        calibrator: LambdaCalibrator | None = None,
        max_iter: int = 20,
        min_features: int = 1,
        min_selected: int = 1,
        random_state: int | None = 42,
        progress: bool = True,
        reuse_null_draws: bool = True,
    ):
        self.model = model
        self.calibrator = calibrator
        self.max_iter = max_iter
        self.min_features = min_features
        self.min_selected = min_selected
        self.random_state = random_state
        self.progress = progress
        self.reuse_null_draws = reuse_null_draws

        self.supports_: Optional[list] = None
        self.n_selected_: Optional[np.ndarray] = None
        self.lambdas_: Optional[np.ndarray] = None
        self.converged_: Optional[np.ndarray] = None
        self.refit_models_: Optional[list] = None
        self.scalers_: Optional[list] = None
        self.results_: Optional[pd.DataFrame] = None
        self.remaining_features_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None
        self.feature_names_in_: Optional[np.ndarray] = None
        self.train_predictions_: Optional[np.ndarray] = None
        self.similarity_matrix_: Optional[np.ndarray] = None
        self.is_fitted_: bool = False

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(model={self.model!r}, "
            f"calibrator={self.calibrator!r}, max_iter={self.max_iter}, "
            f"min_features={self.min_features}, min_selected={self.min_selected}, "
            f"random_state={self.random_state!r}, progress={self.progress}, "
            f"reuse_null_draws={self.reuse_null_draws})"
        )

    def _validate_params(self) -> None:
        if not isinstance(self.model, _BasePicModel):
            raise TypeError(
                f"model must be a picpy model; got {type(self.model).__name__}."
            )
        expected = _CALIBRATORS.get(type(self.model.family))
        if expected is None:
            raise ValueError("Only Gaussian and Binomial families are supported.")
        if self.calibrator is not None:
            if not isinstance(self.calibrator, expected):
                raise TypeError(
                    f"Expected {expected.__name__}; got {type(self.calibrator).__name__}."
                )
            if self.calibrator.fit_intercept != self.model.fit_intercept:
                raise ValueError(
                    "calibrator.fit_intercept must match model.fit_intercept."
                )
        for name in ("max_iter", "min_features", "min_selected"):
            value = getattr(self, name)
            if (
                not isinstance(value, Integral)
                or isinstance(value, (bool, np.bool_))
                or value < 1
            ):
                raise ValueError(f"{name} must be an integer >= 1; got {value!r}.")
        if not isinstance(self.progress, (bool, np.bool_)):
            raise TypeError("progress must be a boolean.")
        if not isinstance(self.reuse_null_draws, (bool, np.bool_)):
            raise TypeError("reuse_null_draws must be a boolean.")

    def fit(self, X, y, X_val=None, y_val=None) -> "RecursiveSelection":
        """Run recursive PIC selection with mandatory unpenalized GLM refitting."""
        self._validate_params()

        feature_names = X.columns.to_numpy() if isinstance(X, pd.DataFrame) else None
        device = getattr(self.model, "device", None)
        dtype = getattr(self.model, "dtype", None)
        X, y = check_Xy(
            X, y, y_kind=self.model.family.y_kind, device=device, dtype=dtype
        )
        if device is None and is_torch(X):
            device = X.device

        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must be provided together.")
        if X_val is None:
            X_val, y_val = X, y
            evaluation = "train"
        else:
            X_val, y_val = check_Xy(
                X_val,
                y_val,
                y_kind=self.model.family.y_kind,
                device=device if is_torch(X) else None,
                dtype=dtype,
            )
            if X_val.shape[1] != X.shape[1]:
                raise ValueError("X_val must have the same number of features as X.")
            evaluation = "validation"

        # The sklearn refits and the metrics work on small host arrays.
        y_host = _host_response(y, self.model.family)
        y_val_host = _host_response(y_val, self.model.family)

        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = feature_names

        remaining = np.arange(self.n_features_in_)
        supports, lambdas, converged, records, train_predictions = [], [], [], [], []
        refit_models, scalers = [], []
        rng = np.random.default_rng(self.random_state)

        calibrator = self.calibrator
        if calibrator is None:
            calibrator = _CALIBRATORS[type(self.model.family)](
                method="analytical", fit_intercept=self.model.fit_intercept
            )

        null_scores, shared_cal = None, None
        if self.reuse_null_draws and calibrator.method != "analytical":
            shared_cal = copy.deepcopy(calibrator)
            if hasattr(shared_cal, "random_state"):
                shared_cal.random_state = int(rng.integers(np.iinfo(np.int32).max))
            null_scores = shared_cal.compute_column_scores(_fit_scaler(X)[0])

        params = {**self.model.get_params(), "verbosity": 0, "refit": False}
        iterator = tqdm(
            range(self.max_iter), desc="Recursive Selection", disable=not self.progress
        )

        for iteration in iterator:
            if len(remaining) < self.min_features:
                break

            X_scaled, scaler = _fit_scaler(take_columns(X, remaining))

            if null_scores is not None:
                lambda_ = shared_cal.lambda_from_scores(null_scores, remaining)
            else:
                cal = copy.deepcopy(calibrator)
                if hasattr(cal, "random_state"):
                    cal.random_state = int(rng.integers(np.iinfo(np.int32).max))
                lambda_ = cal.compute(X_scaled)
            pic = type(self.model)(**{**params, "lambda_": lambda_, "refit": False})

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                pic.fit(X_scaled, y)

            if not pic.converged_:
                warnings.warn(
                    f"PIC did not converge at iteration {iteration + 1}.",
                    ConvergenceWarning,
                    stacklevel=2,
                )

            selected_local = flatnonzero(pic.coef_)
            if len(selected_local) < self.min_selected:
                break

            selected = remaining[selected_local]
            X_refit = to_numpy(take_columns(X_scaled, selected_local))
            X_val_refit = to_numpy(
                take_columns(
                    _transform(take_columns(X_val, remaining), scaler), selected_local
                )
            )

            glm, metrics = _refit_and_evaluate(
                X_refit,
                y_host,
                X_val_refit,
                y_val_host,
                family=self.model.family,
                fit_intercept=self.model.fit_intercept,
            )

            if isinstance(self.model.family, Binomial):
                pred_train = glm.predict_proba(X_refit)[:, 1]
            else:
                pred_train = glm.predict(X_refit)

            train_predictions.append(pred_train)

            supports.append(selected.copy())
            lambdas.append(lambda_)
            converged.append(bool(pic.converged_))
            refit_models.append(glm)
            scaler_refit = StandardScaler()
            scaler_refit.fit(to_numpy(take_columns(X, selected)))
            scalers.append(scaler_refit)

            records.append(
                {
                    "iteration": iteration + 1,
                    "lambda": lambda_,
                    "n_selected": len(selected),
                    "n_remaining": len(remaining) - len(selected),
                    "converged": bool(pic.converged_),
                    "evaluation": evaluation,
                    **metrics,
                }
            )

            remaining = np.delete(remaining, selected_local)
            iterator.set_postfix(selected=len(selected), remaining=len(remaining))

        self.supports_ = supports
        self.n_selected_ = np.array([len(s) for s in supports], dtype=int)
        self.lambdas_ = np.array(lambdas, dtype=float)
        self.converged_ = np.array(converged, dtype=bool)
        self.refit_models_ = refit_models
        self.scalers_ = scalers
        self.results_ = pd.DataFrame(records)
        self.remaining_features_ = remaining
        self.is_fitted_ = True
        self.train_predictions_ = (
            np.column_stack(train_predictions)
            if train_predictions
            else np.empty((len(y_host), 0))
        )
        self.similarity_matrix_ = self._compute_similarity(y_host)
        return self

    def get_support(self, iteration: int) -> np.ndarray:
        """Return original feature indices selected at a given iteration (1-based)."""
        self._check_is_fitted()
        if not 1 <= iteration <= len(self.supports_):
            raise ValueError(f"iteration must be between 1 and {len(self.supports_)}.")
        return self.supports_[iteration - 1].copy()

    def summary(self) -> pd.DataFrame:
        """Return iteration-level results."""
        self._check_is_fitted()
        return self.results_.copy()

    def predict(self, X, iteration: int = 1) -> np.ndarray:
        """Predict using the unpenalized GLM from a given iteration."""
        self._check_is_fitted()
        support, glm, scaler = self._get_iteration(iteration)
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have shape (n_samples, {self.n_features_in_}).")
        return glm.predict(scaler.transform(X[:, support]))

    def predict_proba(self, X, iteration: int = 1) -> np.ndarray:
        """Predict class probabilities for Binomial models."""
        self._check_is_fitted()
        if not isinstance(self.model.family, Binomial):
            raise AttributeError("predict_proba is only available for Binomial models.")
        support, glm, scaler = self._get_iteration(iteration)
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have shape (n_samples, {self.n_features_in_}).")
        return glm.predict_proba(scaler.transform(X[:, support]))

    def _compute_similarity(self, y: np.ndarray) -> np.ndarray:
        """Compute pairwise similarity of unpenalized GLMs on training data."""
        P = self.train_predictions_
        K = P.shape[1]

        if K == 0:
            return np.empty((0, 0))

        if isinstance(self.model.family, Binomial):
            predictions = (P >= 0.5).astype(int)
            errors = (predictions != y[:, None]).astype(int)

            intersection = errors.T @ errors
            sizes = errors.sum(axis=0)
            union = sizes[:, None] + sizes[None, :] - intersection

            return np.divide(
                intersection,
                union,
                out=np.ones((K, K), dtype=float),
                where=union > 0,
            )

        if isinstance(self.model.family, Gaussian):
            squared_norms = np.sum(P**2, axis=0)
            squared_distances = (
                squared_norms[:, None] + squared_norms[None, :] - 2 * P.T @ P
            )

            rmse = np.sqrt(np.maximum(squared_distances, 0) / len(y))
            scale = np.std(y)

            if scale == 0:
                raise ValueError(
                    "Cannot normalize RMSE: training response has zero variance."
                )

            D = rmse / scale
            np.fill_diagonal(D, 0.0)
            return D

        raise ValueError("Unsupported family.")

    def plot_performance(self, metric: str = "accuracy", ax=None):
        """Plot predictive performance across recursive iterations."""
        self._check_is_fitted()
        if metric not in self.results_.columns:
            raise ValueError(f"Unknown metric {metric!r}.")
        if self.results_.empty:
            raise ValueError("No supports were selected.")
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))
        ax.plot(self.results_["iteration"], self.results_[metric], marker="o")
        ax.set(xlabel="Iteration", ylabel=metric, title=f"Recursive PIC: {metric}")
        ax.grid(alpha=0.3)
        ax.figure.tight_layout()
        return ax

    def plot_prediction_similarity(self, figsize=(8, 7), ax=None):
        """Plot pairwise similarity of recursive GLMs on training data."""
        self._check_is_fitted()

        M = self.similarity_matrix_
        if M.size == 0:
            raise ValueError("No supports were selected.")

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        if isinstance(self.model.family, Binomial):
            title = "Error Jaccard similarity (training)"
            cmap, vmin, vmax = "viridis", 0, 1
        else:
            title = "Normalized prediction RMSE (training)"
            cmap, vmin, vmax = "viridis", 0, None

        im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.figure.colorbar(im, ax=ax, label=title)

        ticks = np.arange(M.shape[0])
        ax.set_xticks(ticks, ticks + 1)
        ax.set_yticks(ticks, ticks + 1)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Iteration")
        ax.set_title(title)

        ax.figure.tight_layout()
        return ax

    def plot_supports(self, ax=None):
        """Plot the disjoint supports in the original feature space."""
        self._check_is_fitted()
        if not self.supports_:
            raise ValueError("No supports were selected.")
        matrix = np.zeros((len(self.supports_), self.n_features_in_), dtype=bool)
        for i, support in enumerate(self.supports_):
            matrix[i, support] = True
        selected = np.flatnonzero(matrix.any(axis=0))
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 5))
        ax.imshow(
            matrix[:, selected],
            aspect="auto",
            cmap="Greys",
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )
        ax.set(
            xlabel="Selected feature (original index)",
            ylabel="Iteration",
            title="Recursive PIC: disjoint supports",
        )
        step = max(1, int(np.ceil(len(selected) / 40)))
        ticks = np.arange(0, len(selected), step)
        ax.set_xticks(ticks, selected[ticks], rotation=90)
        ax.set_yticks(
            np.arange(len(self.supports_)), np.arange(1, len(self.supports_) + 1)
        )
        ax.figure.tight_layout()
        return ax

    def _get_iteration(self, iteration):
        if not 1 <= iteration <= len(self.supports_):
            raise ValueError(f"iteration must be between 1 and {len(self.supports_)}.")
        i = iteration - 1
        return self.supports_[i], self.refit_models_[i], self.scalers_[i]

    def _check_is_fitted(self):
        if not self.is_fitted_:
            raise ValueError("Call 'fit' before accessing results.")


def _host_response(y, family):
    """Response as a NumPy array for sklearn (int labels for Binomial)."""
    y = to_numpy(y)
    return y.astype(int) if isinstance(family, Binomial) else y.astype(float)


def _fit_scaler(X):
    """Standardize *X* column-wise (ddof=0) in its own backend.

    NumPy uses sklearn's ``StandardScaler`` (historical behaviour); tensors
    are standardized on their device.  Returns ``(X_scaled, scaler)`` where
    *scaler* is what :func:`_transform` needs.
    """
    if is_torch(X):
        import torch

        mean = X.mean(dim=0)
        std = X.std(dim=0, correction=0)
        scale = torch.where(std > 0, std, torch.ones_like(std))
        return (X - mean) / scale, (mean, scale)
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler


def _transform(X, scaler):
    """Apply a scaler returned by :func:`_fit_scaler`."""
    if isinstance(scaler, tuple):
        mean, scale = scaler
        return (X - mean) / scale
    return scaler.transform(X)


def _refit_and_evaluate(X, y, X_val, y_val, family, fit_intercept=True):
    """Fit an unpenalized sklearn GLM and evaluate its predictions."""
    if isinstance(family, Binomial):
        glm = SkLogisticRegression(
            C=np.inf, fit_intercept=fit_intercept, max_iter=2000, tol=1e-8
        )
        with warnings.catch_warnings():
            # scikit-learn >= 1.8 recommends C=np.inf for an unpenalized fit
            # but still warns that it "will ignore the C parameter".
            warnings.filterwarnings(
                "ignore", message="Setting penalty=None will ignore", category=UserWarning
            )
            glm.fit(X, y)
        proba = glm.predict_proba(X_val)[:, 1]
        pred = (proba >= 0.5).astype(int)
        metrics = {
            "accuracy": accuracy_score(y_val, pred),
            "auc": (
                roc_auc_score(y_val, proba) if len(np.unique(y_val)) == 2 else np.nan
            ),
            "f1": f1_score(y_val, pred, zero_division=0),
            "log_loss": log_loss(y_val, proba, labels=[0, 1]),
        }
        return glm, metrics

    if isinstance(family, Gaussian):
        glm = SkLinearRegression(fit_intercept=fit_intercept)
        glm.fit(X, y)
        pred = glm.predict(X_val)
        mse = mean_squared_error(y_val, pred)
        metrics = {
            "mse": mse,
            "rmse": np.sqrt(mse),
            "r2": r2_score(y_val, pred),
        }
        return glm, metrics

    raise ValueError(f"Unsupported family: {type(family).__name__}")
