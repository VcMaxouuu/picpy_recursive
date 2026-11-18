from warnings import warn

import numpy as np

from ._array_api import as_float_array, get_namespace, is_torch, to_backend, to_numpy


def _as_2d_float_array(X, name: str = "X", dtype=None):
    """Convert *X* to a 2-D float array, reshaping a 1-D input if necessary.

    NumPy-like inputs become ``float64`` NumPy arrays (or *dtype*); tensors
    stay tensors on their device.
    """
    if not is_torch(X):
        X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError(
            f"{name} must be 2-D (n_samples, n_features). Got shape {tuple(X.shape)}."
        )
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError(f"{name} must have positive shape. Got {tuple(X.shape)}.")
    return as_float_array(X, dtype)


def _as_1d_array(y, name: str = "y"):
    """Convert *y* to a 1-D float array, squeezing a degenerate 2-D input."""
    if not is_torch(y):
        y = np.asarray(y)
    if y.ndim == 2 and (y.shape[1] == 1 or y.shape[0] == 1):
        y = y.reshape(-1)
    if y.ndim != 1:
        raise ValueError(f"{name} must be 1-D (n_samples,). Got shape {tuple(y.shape)}.")
    if y.shape[0] == 0:
        raise ValueError(f"{name} must have positive length. Got {tuple(y.shape)}.")
    return as_float_array(y)


def _check_finite(arr, name: str) -> None:
    """Raise ValueError if *arr* contains NaN or Inf."""
    xp = get_namespace(arr)
    if not bool(xp.all(xp.isfinite(arr))):
        raise ValueError(f"{name} contains NaN or Inf values.")


def check_X(X, n_features: int | None = None, device=None, dtype=None):
    """Validate the design matrix.

    Parameters
    ----------
    X : array-like or torch.Tensor of shape (n_samples, n_features)
        Design matrix.
    n_features : int or None
        Expected number of columns (e.g. the number seen during fit). When
        provided, the number of columns of *X* is checked against it.
    device : None, str or torch.device, default=None
        Compute device.  ``None`` keeps *X* in its own backend; otherwise *X*
        is converted to a ``torch.Tensor`` on that device.
    dtype : None, "float32" or "float64", default=None
        Precision of the returned array (see
        :func:`picpy.utils._array_api.as_float_array`).
    """
    X = _as_2d_float_array(to_backend(X, device), "X", dtype=dtype)
    _check_finite(X, "X")
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError(
            f"X has {X.shape[1]} features, but the model was fitted with "
            f"{n_features} features."
        )
    return X


def check_standardized(X, caller: str, tol: float = 1e-4) -> None:
    """Warn when the columns of *X* are not zero-mean, unit-variance (ddof=0).

    The regularization parameter is calibrated on a standardized design, so
    the same standardized *X* must be used for calibration and fitting.
    In single precision the tolerance is relaxed to the float32 resolution.
    """
    xp = get_namespace(X)
    if is_torch(X) and X.dtype.itemsize < 8:
        tol = max(tol, 1e-3)
    max_mean = float(xp.max(xp.abs(xp.mean(X, axis=0))))
    max_std = float(xp.max(xp.abs(xp.std(X, axis=0, correction=0) - 1.0)))
    if max_mean >= tol or max_std >= tol:
        warn(
            f"{caller}: design matrix X does not appear to be standardized. "
            f"max|mean| = {max_mean:.3e}, max|std - 1| = {max_std:.3e}. "
            "Standardize X once, then use that same X to calibrate lambda_ "
            "and to fit the model.",
            stacklevel=3,
        )


def check_y(
    y,
    n_samples: int | None = None,
    y_kind: str = "continuous",
    like=None,
):
    """Validate the response vector according to its distributional type.

    Parameters
    ----------
    y : array-like or torch.Tensor of shape (n_samples,)
        Response vector.  A 2-D input with a single column (or row) is
        flattened.
    n_samples : int or None
        Expected number of observations. When provided, the length of *y*
        is checked against this value.
    y_kind : {'continuous', 'binary'}
        Type of response variable:

        - ``'continuous'`` — real-valued outcomes (Gaussian).
        - ``'binary'`` — 0/1 labels (Binomial); both classes must be present.
    like : array or None
        Validated design matrix.  When it is a tensor, *y* is returned as a
        tensor with the same device and dtype.

    Returns
    -------
    y : ndarray or torch.Tensor of shape (n_samples,)
        Validated response array.  NumPy binary responses are ``int``;
        tensor responses share the floating dtype of *X*.

    Raises
    ------
    ValueError
        If *y* does not satisfy the constraints implied by *y_kind*.
    """
    if like is not None and is_torch(like):
        y = to_backend(y, like.device)
    y = _as_1d_array(y, "y")
    if like is not None and is_torch(like):
        y = y.to(like.dtype)
    _check_finite(y, "y")

    if n_samples is not None and y.shape[0] != n_samples:
        raise ValueError(
            f"Length of y ({y.shape[0]}) does not match the number of rows "
            f"in X ({n_samples})."
        )

    if y_kind == "binary":
        uniq = np.unique(to_numpy(y))
        if not np.array_equal(uniq, [0.0, 1.0]):
            raise ValueError(
                "Binary response y must contain both classes 0 and 1 and no "
                f"other value. Found: {uniq}."
            )
        if not is_torch(y):
            y = y.astype(int)

    elif y_kind != "continuous":
        raise ValueError(
            f"Unknown y_kind='{y_kind}'. " "Expected one of: 'continuous', 'binary'."
        )

    return y


def check_Xy(X, y, y_kind: str, device=None, dtype=None):
    """Jointly validate and preprocess the design matrix and response.

    With ``device`` set, both are moved to that torch device; a tensor *X*
    also pulls *y* onto its device and dtype.
    """
    X = check_X(X, device=device, dtype=dtype)
    y = check_y(y, n_samples=X.shape[0], y_kind=y_kind, like=X)
    return X, y
