import numpy as np

from ..utils._array_api import get_namespace
from ._base import Family
from .links import Link, IdentityLink, SquareRootLink


class Gaussian(Family):
    """Gaussian (normal) family with identity link and square-root stabilization.

    Loss:

    .. math::

        \\ell(y, \\hat\\mu) = \\frac{1}{n}\\sum_i (y_i - \\hat\\mu_i)^2

    with :math:`\\phi = \\sqrt{\\cdot}`.
    """

    g: Link = IdentityLink()
    phi: Link = SquareRootLink()

    def raw_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        r = y_true - y_pred
        return get_namespace(r).mean(r * r, axis=0)

    def raw_loss_derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        n = y_true.shape[0]
        return (2.0 / n) * (y_pred - y_true)
