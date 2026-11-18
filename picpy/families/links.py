"""Link functions for the picpy families.

Each :class:`Link` maps a linear predictor :math:`\\eta` to the response
scale and provides its derivative and inverse. They are used as the
mean-function link *g* and the variance-stabilizing transform *phi* inside
:class:`~picpy.families.Family` subclasses.
"""

import numpy as np

from ..utils._array_api import get_namespace


class Link:
    """Abstract base class for a link function.

    A link :math:`g` operates on the linear predictor :math:`\\eta`:

    - ``g(eta)`` — forward map: linear predictor → response scale.
    - ``g.derivative(eta)`` — derivative :math:`g'(\\eta)`.
    - ``g.inverse(mu)`` — inverse map: response scale → linear predictor.
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply the link function."""
        raise NotImplementedError

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the derivative of the link function."""
        raise NotImplementedError

    def inverse(self, x: np.ndarray) -> np.ndarray:
        """Apply the inverse of the link function."""
        raise NotImplementedError

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return self.__str__()


class IdentityLink(Link):
    """Identity link: :math:`g(\\eta) = \\eta`."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x

    def derivative(self, x: np.ndarray) -> float:
        return 1.0

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return x


class SquareRootLink(Link):
    """Square-root link: :math:`g(\\eta) = \\sqrt{\\eta}`."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return get_namespace(x).sqrt(x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 0.5 / get_namespace(x).sqrt(x)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return x**2


class LogisticLink(Link):
    """Logistic (sigmoid) link: :math:`g(\\eta) = (1 + e^{-\\eta})^{-1}`."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + get_namespace(x).exp(-x))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        p = self(x)
        return p * (1.0 - p)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        xp = get_namespace(x)
        if bool(xp.any((x <= 0) | (x >= 1))):
            raise ValueError(
                "LogisticLink: inverse (logit) is undefined outside the open interval (0, 1)."
            )
        return xp.log(x / (1.0 - x))
