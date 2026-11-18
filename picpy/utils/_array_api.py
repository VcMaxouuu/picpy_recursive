"""Array-backend helpers: run the same numerical code on NumPy or PyTorch.

picpy follows the Python Array API standard through ``array_api_compat``:
every numerical routine looks up the namespace ``xp`` of its inputs and only
calls functions of that namespace.  NumPy arrays therefore stay on the CPU
and are processed exactly as before, while ``torch.Tensor`` inputs are
processed by PyTorch on their own device (CPU or GPU) without any copy to
the host.

Only small metadata (selected indices, scalars, diagnostics) is brought
back to NumPy.
"""

from __future__ import annotations

import numpy as np
import array_api_compat
from array_api_compat import array_namespace, is_numpy_namespace, is_torch_array

__all__ = [
    "get_namespace",
    "is_torch",
    "to_numpy",
    "to_backend",
    "as_float_array",
    "take_columns",
    "flatnonzero",
    "quantile",
    "make_rng",
    "rademacher01",
    "standard_normal",
]


def get_namespace(*arrays):
    """Array API namespace shared by *arrays*.

    ``None`` and Python scalars are ignored; NumPy is used when no array is
    given.
    """
    arrays = [
        a for a in arrays if a is not None and not isinstance(a, (int, float, bool))
    ]
    if not arrays:
        return array_api_compat.numpy
    return array_namespace(*arrays)


def is_torch(x) -> bool:
    """True when *x* is a ``torch.Tensor``."""
    return is_torch_array(x)


def to_numpy(x):
    """Return *x* as a NumPy array (copied to the host when needed)."""
    if is_torch_array(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "A device was requested but PyTorch is not installed. "
            "Install torch, or leave device=None to compute with NumPy."
        ) from exc
    return torch


def to_backend(x, device=None):
    """Move *x* to the compute backend selected by *device*.

    Parameters
    ----------
    x : array-like or torch.Tensor
    device : None, str or torch.device
        ``None`` keeps *x* where it is (NumPy stays NumPy, a tensor stays on
        its device).  Anything else (``"cuda"``, ``"cuda:1"``, ``"cpu"``,
        ``torch.device``) converts *x* to a ``torch.Tensor`` on that device.
    """
    if device is None:
        return x
    torch = _torch()
    if is_torch_array(x):
        return x.to(device)
    arr = np.asarray(x)
    if arr.dtype.kind in "biu":
        return torch.as_tensor(arr, device=device)
    return torch.as_tensor(arr).to(device)


_PRECISIONS = {"float32": "float32", "float64": "float64", "single": "float32", "double": "float64"}


def resolve_precision(dtype) -> str | None:
    """Normalize a precision request to ``"float32"``, ``"float64"`` or None.

    Accepts None, the strings ``"float32"``/``"float64"`` and NumPy or torch
    float dtypes.
    """
    if dtype is None:
        return None
    name = str(dtype).replace("torch.", "").replace("<class 'numpy.", "").rstrip("'>")
    if name not in _PRECISIONS:
        raise ValueError(f"dtype must be 'float32', 'float64' or None; got {dtype!r}.")
    return _PRECISIONS[name]


def as_float_array(X, dtype=None):
    """Floating-point version of *X* in its own namespace.

    Parameters
    ----------
    X : ndarray or torch.Tensor
    dtype : None, "float32" or "float64"
        Requested precision.  ``None`` keeps the historical behaviour: NumPy
        inputs are cast to ``float64``; tensors keep ``float32``/``float64``
        and other tensor dtypes (int, bool, half) are promoted to ``float32``.
    """
    precision = resolve_precision(dtype)
    if is_torch_array(X):
        import torch

        if precision is not None:
            return X.to(getattr(torch, precision))
        if X.dtype in (torch.float32, torch.float64):
            return X
        return X.to(torch.float32)
    return np.asarray(X).astype(precision or float)


def take_columns(X, idx):
    """``X[:, idx]`` for an integer index given as NumPy, list or tensor."""
    if is_torch_array(X):
        import torch

        idx = torch.as_tensor(np.asarray(to_numpy(idx), dtype=np.int64), device=X.device)
        return X.index_select(1, idx)
    return X[:, np.asarray(idx, dtype=int)]


def flatnonzero(x) -> np.ndarray:
    """Indices of the non-zero entries of *x*, always as a NumPy int array."""
    return np.flatnonzero(to_numpy(x).ravel() != 0)


def quantile(x, q: float) -> float:
    """Linear-interpolation quantile (NumPy's default) as a Python float."""
    if is_torch_array(x):
        import torch

        return float(torch.quantile(x, q))
    return float(np.quantile(x, q))


# ---------------------------------------------------------------------------
# Random draws on the data's device
# ---------------------------------------------------------------------------


def make_rng(random_state, like, backend: str = "numpy"):
    """Random generator for Monte Carlo draws used with *like*.

    Parameters
    ----------
    random_state : int, np.random.Generator or None
    like : ndarray or torch.Tensor
        Data the draws are combined with.
    backend : {"numpy", "device"}, default="numpy"
        ``"numpy"`` always draws with ``np.random.default_rng(random_state)``
        and moves the draws to the data's device, so a given seed gives the
        same draws, hence the same lambda, on every backend.  ``"device"``
        uses a ``torch.Generator`` on the tensor's device: faster for very
        large designs, but the draws differ from NumPy's.
    """
    if backend not in ("numpy", "device"):
        raise ValueError(f"rng_backend must be 'numpy' or 'device'; got {backend!r}.")
    if backend == "numpy" or not is_torch_array(like):
        return np.random.default_rng(random_state)
    import torch

    if isinstance(random_state, np.random.Generator):
        seed = int(random_state.integers(np.iinfo(np.int64).max))
    elif random_state is None:
        seed = int(np.random.default_rng().integers(np.iinfo(np.int64).max))
    else:
        seed = int(random_state)
    gen = torch.Generator(device=like.device)
    gen.manual_seed(seed)
    return gen


def rademacher01(rng, shape, like):
    """Bernoulli(1/2) draws in {0, 1} with the dtype/device of *like*."""
    if isinstance(rng, np.random.Generator):
        draws = rng.binomial(1, 0.5, size=shape)
        if is_torch_array(like):
            import torch

            return torch.as_tensor(draws, device=like.device).to(like.dtype)
        return draws
    import torch

    out = torch.randint(0, 2, shape, generator=rng, device=like.device)
    return out.to(like.dtype)


def standard_normal(rng, shape, like):
    """Standard normal draws with the dtype/device of *like*."""
    if isinstance(rng, np.random.Generator):
        draws = rng.standard_normal(shape)
        if is_torch_array(like):
            import torch

            return torch.as_tensor(draws, device=like.device).to(like.dtype)
        return draws
    import torch

    return torch.randn(shape, generator=rng, device=like.device, dtype=like.dtype)


def is_numpy(xp) -> bool:
    """True when *xp* is the NumPy namespace."""
    return is_numpy_namespace(xp)
