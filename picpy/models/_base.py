"""Base class for all models presented in the PIC framework."""

import inspect
from numbers import Integral, Real
from typing import Optional
from warnings import warn

import numpy as np
from array_api_compat import device as _device
from pandas import DataFrame

from ..exceptions import ConvergenceWarning
from ..families import Family
from ..penalties import Penalty, L1Penalty
from ..solvers import fista
from ..utils import check_Xy, check_X, check_standardized, TrainingLogger
from ..utils._array_api import (
    flatnonzero,
    get_namespace,
    is_torch,
    resolve_precision,
    take_columns,
    to_numpy,
)


class _BasePicModel:
    """Abstract base class for picpy models.

    Encapsulates the shared fitting pipeline common to all models.  Follows
    the scikit-learn estimator conventions: hyperparameters are set in the
    constructor, :meth:`fit` only takes data, and :meth:`get_params` /
    :meth:`set_params` are available.

    Parameters
    ----------
    family : Family
        Distributional family.
    lambda_ : float
        Regularization parameter (> 0). Should be computed by a
        `LambdaCalibrator` subclass on the same standardized *X* that is
        passed to :meth:`fit`.
    penalty : Penalty, default=L1Penalty()
        Sparsity-inducing penalty applied at the final regularization step.
        L1 is always used for the warm-up steps regardless of this choice.
    fit_intercept : bool, default=True
        Whether to estimate an unpenalized intercept term.
    refit : bool, default=False
        Whether to run an unpenalized refit on the selected support after
        the regularization path.  Reduces estimation bias.
    tol : float, default=1e-6
        Relative convergence tolerance for FISTA at the final path step and
        for the refit.
    max_iter : int, default=500
        Maximum number of FISTA iterations per path step (and for the refit).
    verbosity : int, default=0
        Amount of output printed to stdout during :meth:`fit`:

        - ``0`` — silent.
        - ``1`` — high-level progress (per-step convergence, final summary).
        - ``2`` — adds per-iteration objective values.
    log_interval : int, default=5
        At ``verbosity=2``, print one iteration line every *log_interval*
        iterations.
    device : None, str or torch.device, default=None
        Where the solver runs.  ``None`` computes in the backend of the
        input: NumPy arrays on the CPU with NumPy, ``torch.Tensor`` inputs
        with PyTorch on their own device.  ``"cuda"`` (or any torch device)
        converts the inputs to tensors on that device first.
    dtype : {"float64", "float32"}, default="float64"
        Precision of the solver, whatever the input dtype.  ``"float64"``
        reproduces the NumPy results (also on a GPU).  ``"float32"`` halves
        memory and is about twice as fast on a GPU, but the non-convex
        penalties (SCAD, MCP) may then select slightly different supports.

    Notes
    -----
    Fitted arrays follow the input: ``coef_`` is a NumPy array when *X* was
    a NumPy array (even if the solver ran on a GPU) and a tensor on the
    compute device when *X* was a tensor.  Prediction methods return arrays
    in the namespace of the *X* they receive.

    Attributes
    ----------
    coef_ : ndarray or torch.Tensor of shape (n_features,)
        Fitted coefficients.
    intercept_ : float or None
        Fitted intercept.  None when *fit_intercept=False*.
    loss_ : float
        Smooth (stabilized) loss at the fitted coefficients.
    objective_ : float
        Penalized objective ``loss_ + penalty(coef_, lambda_)`` at the fitted
        coefficients.  With ``refit=True`` it is evaluated at the refitted
        coefficients, which do not minimize it.
    n_iter_ : int
        Number of FISTA iterations of the last solver run (the final path
        step, or the refit when ``refit=True``).
    converged_ : bool
        Whether the final path step (and the refit, if any) reached *tol*.
    """

    def __init__(
        self,
        family: Family,
        lambda_: float,
        penalty: Penalty = L1Penalty(),
        fit_intercept: bool = True,
        refit: bool = False,
        tol: float = 1e-6,
        max_iter: int = 500,
        verbosity: int = 0,
        log_interval: int = 5,
        device=None,
        dtype: str = "float64",
    ):
        self.family = family
        self.lambda_ = lambda_
        self.penalty = penalty
        self.fit_intercept = fit_intercept
        self.refit = refit
        self.tol = tol
        self.max_iter = max_iter
        self.verbosity = verbosity
        self.log_interval = log_interval
        self.device = device
        self.dtype = dtype

        # Set during fit
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        self.loss_: Optional[float] = None
        self.objective_: Optional[float] = None
        self.n_iter_: Optional[int] = None
        self.converged_: Optional[bool] = None
        self.n_samples_in_: Optional[int] = None
        self.n_features_in_: Optional[int] = None
        self.feature_names_in_: Optional[np.ndarray] = None
        self.is_fitted_: bool = False

        # Per-step solver diagnostics
        self._fit_info: Optional[dict] = None

    # ------------------------------------------------------------------
    # Parameters (scikit-learn estimator API)
    # ------------------------------------------------------------------

    @classmethod
    def _get_param_names(cls) -> list[str]:
        """Constructor argument names of the concrete class."""
        signature = inspect.signature(cls.__init__)
        return [name for name in signature.parameters if name != "self"]

    def get_params(self, deep: bool = True) -> dict:
        """Get the constructor parameters of this model.

        Parameters
        ----------
        deep : bool, default=True
            Accepted for scikit-learn compatibility; has no effect.

        Returns
        -------
        dict
            Parameter names mapped to their values.
        """
        return {name: getattr(self, name) for name in self._get_param_names()}

    def set_params(self, **params) -> "_BasePicModel":
        """Set constructor parameters of this model.

        Returns
        -------
        self : object
        """
        valid = self._get_param_names()
        for name, value in params.items():
            if name not in valid:
                raise ValueError(
                    f"Invalid parameter {name!r} for {type(self).__name__}. "
                    f"Valid parameters are: {valid}."
                )
            setattr(self, name, value)
        return self

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({params})"

    def _validate_params(self) -> None:
        if not isinstance(self.lambda_, Real) or not np.isfinite(self.lambda_):
            raise ValueError(f"lambda_ must be a finite number; got {self.lambda_!r}.")
        if self.lambda_ <= 0:
            raise ValueError(f"lambda_ must be > 0; got {self.lambda_!r}.")
        if not isinstance(self.penalty, Penalty):
            raise TypeError(
                f"penalty must be a Penalty instance; got {type(self.penalty).__name__}."
            )
        if not self.tol > 0:
            raise ValueError(f"tol must be > 0; got {self.tol!r}.")
        if not isinstance(self.max_iter, Integral) or self.max_iter < 1:
            raise ValueError(
                f"max_iter must be a positive integer; got {self.max_iter!r}."
            )
        resolve_precision(self.dtype)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def selected_features_(self) -> np.ndarray:
        """Indices of features with non-zero fitted coefficients.

        Returns
        -------
        ndarray of int
            Zero-based column indices of the selected features.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        self._check_is_fitted()
        return flatnonzero(self.coef_)

    @property
    def n_selected_(self) -> int:
        """Number of features with non-zero fitted coefficients.

        Returns
        -------
        int

        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        self._check_is_fitted()
        return len(flatnonzero(self.coef_))

    @property
    def selected_feature_names_(self) -> np.ndarray:
        """Names of features with non-zero fitted coefficients.

        Returns
        -------
        ndarray of str

        Raises
        ------
        ValueError
            If the model has not been fitted yet, or if no feature names
            are available (input was a plain array, not a DataFrame).
        """
        self._check_is_fitted()
        if self.feature_names_in_ is None:
            raise ValueError(
                "Feature names are not available. Fit the model with a "
                "pandas DataFrame to enable this property."
            )
        return self.feature_names_in_[self.selected_features_]

    @property
    def coef_table(self) -> DataFrame:
        """Return a tidy summary of estimated coefficients.

        One row per feature.  Selected features (non-zero coefficients) appear
        first, sorted by descending absolute coefficient value, followed by the
        unselected features.

        Returns
        -------
        pd.DataFrame
            Columns:

            - ``feature`` — feature name (column index when no names are
              available).
            - ``coef`` — estimated coefficient.
            - ``abs_coef`` — absolute value of the coefficient.
            - ``selected`` — True when the coefficient is non-zero.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        self._check_is_fitted()

        if self.feature_names_in_ is not None:
            names = list(self.feature_names_in_)
        else:
            names = list(range(self.n_features_in_))

        coef = to_numpy(self.coef_)
        abs_coef = np.abs(coef)
        selected = coef != 0

        df = DataFrame(
            {
                "coef": coef,
                "abs_coef": abs_coef,
                "selected": selected,
            },
            index=names,
        )

        df.index.name = "feature"
        df = df.sort_values(["selected", "abs_coef"], ascending=[False, False])
        return df

    @property
    def penalty_value(self) -> float:
        """Returns the penalty part of the loss function.

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        self._check_is_fitted()
        return self.penalty.evaluate(self.coef_, self.lambda_)

    def summary(self) -> None:
        """Print a formatted summary of the fitted model.

        Displays two sections:

        1. **Model** — class name, family, penalty, lambda, and fit settings.
        2. **Selected variables** — coefficient table for non-zero features,
           sorted by descending absolute value; intercept appended when fitted.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model.fit(X, y)
        >>> model.summary()
        +---------------------------------------------------+
        |  Model         : LinearRegression                 |
        |  Family        : Gaussian                         |
        |  Penalty       : L1Penalty                        |
        |  lambda        :     0.2254                       |
        |  n             :        200                       |
        |  p             :         20                       |
        |  fit_intercept :       True                       |
        +---------------------------------------------------+
        |  Selected variables (3 / 20)                      |
        |  feature       :       coef                       |
        |  x0            :     2.0821                       |
        |  x1            :    -1.4763                       |
        |  x2            :     0.9912                       |
        |  intercept     :     0.0312                       |
        +---------------------------------------------------+
        """
        self._check_is_fitted()

        inner_width = 49

        def sep() -> str:
            return "+" + "-" * (inner_width + 2) + "+"

        def row(label: str, value: str, marker: str = "") -> str:
            content = f"  {label:<16}: {value:>10}{marker}"
            return f"| {content:<{inner_width}} |"

        def section_header(title: str) -> str:
            content = f"  {title}"
            return f"| {content:<{inner_width}} |"

        lines = [sep()]

        # --- Section 1: Model ---
        lines.append(row("Model", type(self).__name__))
        lines.append(row("Family", type(self.family).__name__))
        lines.append(row("Penalty", type(self.penalty).__name__))
        lines.append(row("lambda", f"{self.lambda_:.4f}"))
        lines.append(row("n", f"{self.n_samples_in_:,}"))
        lines.append(row("p", f"{self.n_features_in_:,}"))
        lines.append(row("fit_intercept", str(self.fit_intercept)))

        # --- Section 2: Selected variables ---
        lines.append(sep())
        n_sel = self.n_selected_
        lines.append(
            section_header(f"Selected variables ({n_sel} / {self.n_features_in_})")
        )
        lines.append(row("feature", "coef"))

        if self.feature_names_in_ is not None:
            names = list(self.feature_names_in_)
        else:
            names = [str(i) for i in range(self.n_features_in_)]

        # selected features sorted by |coef| descending
        coef = to_numpy(self.coef_)
        sel_idx = self.selected_features_
        order = sel_idx[np.argsort(-np.abs(coef[sel_idx]))]
        for i in order:
            lines.append(row(str(names[i]), f"{coef[i]:+.4f}"))

        if self.fit_intercept and self.intercept_ is not None:
            lines.append(row("intercept", f"{self.intercept_:+.4f}"))

        if n_sel == 0:
            lines.append(section_header("  (no features selected)"))

        lines.append(sep())

        print("\n".join(lines))

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_BasePicModel":
        """Fit the model using FISTA on a warm-started path.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Standardized design matrix, the same one used to calibrate
            ``lambda_``.  A warning is issued when the columns do not appear
            to be zero-mean, unit-variance standardized.
        y : array-like of shape (n_samples,)
            Response vector.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self._validate_params()

        # --- Validate data and store meta ---
        feature_names = X.columns.to_numpy() if isinstance(X, DataFrame) else None
        return_numpy = not is_torch(X)
        X, y = check_Xy(
            X, y, y_kind=self.family.y_kind, device=self.device, dtype=self.dtype
        )
        xp = get_namespace(X)
        check_standardized(X, caller=type(self).__name__)
        self.n_samples_in_, self.n_features_in_ = X.shape
        self.feature_names_in_ = feature_names
        logger = TrainingLogger(verbosity=self.verbosity, log_interval=self.log_interval)

        starting_intercept = (
            self.family.starting_intercept(y) if self.fit_intercept else 0.0
        )
        lin_pred_null = xp.full(
            (X.shape[0],), float(starting_intercept), dtype=X.dtype, device=_device(X)
        )
        coef_grad_null, _ = self.family.grad(X, y, lin_pred_null)
        lambda_max_val = float(xp.max(xp.abs(coef_grad_null)))
        lambda_val = self.lambda_

        path_length = 10
        if lambda_val >= lambda_max_val or path_length == 1:
            lambda_path = [lambda_val]
            penalty_path = [self.penalty]
            tol_path = [self.tol]
        else:
            lambda_path = list(
                np.exp(
                    np.linspace(
                        np.log(lambda_max_val),
                        np.log(lambda_val),
                        path_length,
                    )
                )
            )
            penalty_path = [L1Penalty()] * (path_length - 1) + [self.penalty]
            tol_path = [1e-3] * (path_length - 1) + [self.tol]

        logger.start_fitting(
            n_samples=X.shape[0],
            n_features=X.shape[1],
            lambda_reg=lambda_val,
            penalty_name=str(self.penalty),
        )

        coef_prev: Optional[np.ndarray] = None
        intercept_prev: Optional[float] = None
        coef_ = xp.zeros(X.shape[1], dtype=X.dtype, device=_device(X))
        intercept_: Optional[float] = starting_intercept
        training_info: dict = {}

        for idx, (lam, tol, pen) in enumerate(zip(lambda_path, tol_path, penalty_path)):
            logger.log_path_start(
                path_idx=idx,
                total_paths=len(lambda_path),
                lambda_current=lam,
                rel_tol=tol,
                phase="Path",
                penalty_name=type(pen).__name__,
            )

            coef_, intercept_, info = fista(
                X=X,
                y=y,
                family=self.family,
                penalty=pen,
                lambda_reg=lam,
                fit_intercept=self.fit_intercept,
                rel_tol=tol,
                max_iter=self.max_iter,
                logger=logger,
                coef_init=coef_prev,
                intercept_init=intercept_prev,
            )
            coef_prev = coef_
            intercept_prev = intercept_
            training_info[f"step_{idx}"] = info

        info = training_info[f"step_{len(lambda_path) - 1}"]
        if info["status"] != "converged":
            warn(
                f"{type(self).__name__}: solver did not converge at "
                f"lambda_={self.lambda_:.6g} within max_iter={self.max_iter} "
                f"iterations (tol={self.tol:g}). Increase max_iter or tol.",
                ConvergenceWarning,
                stacklevel=2,
            )

        # --- Store penalized solution ---
        self.coef_ = coef_
        self.intercept_ = intercept_
        self.is_fitted_ = True
        self._fit_info = training_info
        statuses = [info["status"]]

        # --- Optional unpenalized refit on selected support ---
        if self.refit:
            support = self.selected_features_
            logger.start_refit(n_selected=len(support))

            if len(support) > 0:
                # L1 at lambda=0 is the identity prox, with no step-size cap.
                X_support = take_columns(X, support)
                support_idx = (
                    xp.asarray(support, device=_device(X)) if is_torch(X) else support
                )
                coef_refit, intercept_refit, info = fista(
                    X=X_support,
                    y=y,
                    family=self.family,
                    penalty=L1Penalty(),
                    lambda_reg=0.0,
                    fit_intercept=self.fit_intercept,
                    rel_tol=self.tol,
                    max_iter=self.max_iter,
                    logger=logger,
                    coef_init=self.coef_[support_idx],
                    intercept_init=self.intercept_,
                )
                coef_full = xp.zeros_like(self.coef_)
                coef_full[support_idx] = coef_refit
                self.coef_ = coef_full
                self.intercept_ = intercept_refit
                self._fit_info["refit"] = info
                statuses.append(info["status"])

                if info["status"] != "converged":
                    warn(
                        f"{type(self).__name__}: unpenalized refit did not "
                        f"converge within max_iter={self.max_iter} iterations. "
                        "The unpenalized problem may have no finite solution "
                        "(e.g. separable classes for LogisticRegression).",
                        ConvergenceWarning,
                        stacklevel=2,
                    )

        # --- Fit results ---
        eta = X @ self.coef_
        if self.fit_intercept:
            eta = eta + self.intercept_
        if (
            self.refit
            and self.family.y_kind == "binary"
            and bool(xp.all((2 * y - 1) * eta > 0))
        ):
            # The relative convergence test can pass while the coefficients
            # still diverge, so separation is checked explicitly.
            warn(
                f"{type(self).__name__}: the classes are perfectly separated "
                "on the selected support, so the unpenalized refit has no "
                "finite solution and coef_ is not meaningful. Use refit=False.",
                ConvergenceWarning,
                stacklevel=2,
            )
        self.loss_ = float(self.family.evaluate(y, eta))
        self.objective_ = self.loss_ + self.penalty_value
        self.n_iter_ = info["n_iter"]
        self.converged_ = all(status == "converged" for status in statuses)
        if return_numpy and is_torch(self.coef_):
            # NumPy in, NumPy out: the solver ran on a device, results come back.
            self.coef_ = to_numpy(self.coef_)

        logger.log_final_results(
            n_selected=self.n_selected_,
            objective_final=self.objective_,
        )
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute the linear predictor :math:`X\\hat\\beta + \\hat b`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Design matrix, standardized like the one passed to :meth:`fit`.

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        self._check_is_fitted()
        X = check_X(X, n_features=self.n_features_in_)
        eta = X @ self._coef_like(X)
        if self.fit_intercept:
            eta = eta + self.intercept_
        return eta

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict fitted values on the response scale.

        Applies the mean-function link *g* to the linear predictor:
        :math:`\\hat y = g(X\\hat\\beta + \\hat b)`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Design matrix, standardized like the one passed to :meth:`fit`.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted values on the response scale.
        """
        return self.family.g(self.decision_function(X))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _coef_like(self, X):
        """``coef_`` in the namespace, device and dtype of *X*."""
        if is_torch(X):
            import torch

            return torch.as_tensor(self.coef_, device=X.device).to(X.dtype)
        return to_numpy(self.coef_)

    def _check_is_fitted(self) -> None:
        if not self.is_fitted_:
            raise ValueError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call 'fit' before using this estimator."
            )
