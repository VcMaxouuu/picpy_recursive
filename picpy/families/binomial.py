import numpy as np

from ..utils._array_api import get_namespace
from ._base import Family
from .links import Link, IdentityLink, LogisticLink

_EPS = 1e-10


def _clip_eps(xp, a) -> float:
    """Probability clipping bound for the dtype of *a*.

    ``1e-10`` in double precision.  In single precision ``1 - 1e-10`` rounds
    to exactly 1, which makes the loss and its gradient divide by zero, so
    the bound is raised to a few float32 ulps.
    """
    return max(_EPS, 4.0 * float(xp.finfo(a.dtype).eps))


class Binomial(Family):
    """Bernoulli family with logistic link for binary classification.

    Loss (variance-stabilized form of the binomial deviance):

    .. math::

        \\ell(y, \\hat\\mu) = \\frac{2}{n}\\sum_i\\left[
                y_i\\sqrt{\\tfrac{1-\\mu_i}{\\mu_i}}
                + (1-y_i)\\sqrt{\\tfrac{\\mu_i}{1-\\mu_i}}
        \\right]

    with :math:`\\phi = \\mathrm{id}` and :math:`g = \\mathrm{logistic}`.
    """

    g: Link = LogisticLink()
    phi: Link = IdentityLink()
    y_kind: str = "binary"

    def raw_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        xp = get_namespace(y_pred)
        eps = _clip_eps(xp, y_pred)
        y_pred = xp.clip(y_pred, eps, 1.0 - eps)
        sqrt_val = xp.sqrt((1.0 - y_pred) / y_pred)
        return 2.0 * xp.mean(
            y_true * sqrt_val + (1.0 - y_true) / sqrt_val,
            axis=0,
        )

    def raw_loss_derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        xp = get_namespace(y_pred)
        eps = _clip_eps(xp, y_pred)
        y_pred = xp.clip(y_pred, eps, 1.0 - eps)
        n = y_true.shape[0]
        return (1.0 / n) * (y_pred - y_true) / (y_pred * (1.0 - y_pred)) ** 1.5
