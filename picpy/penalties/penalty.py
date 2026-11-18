"""Sparsity-inducing penalty functions."""

import numpy as np

from ..utils._array_api import get_namespace
from .proximal import soft_thresholding, mcp_thresholding, scad_thresholding


class Penalty:
    """Abstract base class for a regularization penalty.

    Subclasses must implement :meth:`evaluate` and :meth:`prox`.
    """

    def __str__(self) -> str:
        params = self._param_str()
        return (
            f"{self.__class__.__name__}({params})"
            if params
            else self.__class__.__name__
        )

    def __repr__(self) -> str:
        return self.__str__()

    def _param_str(self) -> str:
        """Return a string representation of penalty-specific parameters."""
        return ""

    @property
    def max_step(self) -> float:
        """Largest step size for which :meth:`prox` is well-defined.

        Solvers should ensure ``step_size < max_step`` before calling
        :meth:`prox`.  For convex penalties (L1, ridge), the prox is
        defined for any positive step size and this returns
        ``float('inf')``.  For non-convex penalties (SCAD, MCP), the
        prox formula has a finite domain of validity and exceeding it
        produces incorrect results (sign flips, division by zero, or
        silently wrong outputs when masks happen to be empty).

        Returns
        -------
        float
            Strict upper bound on admissible step size.  Defaults to
            ``float('inf')`` for convex penalties.
        """
        return float("inf")

    def evaluate(self, parameter: np.ndarray, lambda_: float) -> float:
        """Evaluate the penalty at *parameter*.

        Parameters
        ----------
        parameter : ndarray
            Coefficient vector :math:`\\beta`.
        lambda_ : float
            Regularization parameter.

        Returns
        -------
        float
            Penalty value.
        """
        raise NotImplementedError

    def prox(
        self, parameter: np.ndarray, lambda_: float, step_size: float
    ) -> np.ndarray:
        """Apply the proximal operator of the penalty.

        Solves :math:`\\arg\\min_u \\{t\\lambda P(u) + \\tfrac{1}{2}\\|u - v\\|^2\\}`
        where *step_size* is :math:`t`.

        Parameters
        ----------
        parameter : ndarray
            Input vector :math:`v`.
        lambda_ : float
            Regularization parameter.
        step_size : float
            Step size :math:`t`.

        Returns
        -------
        ndarray
            Result of the proximal map.
        """
        raise NotImplementedError


class L1Penalty(Penalty):
    """Lasso (L1) penalty: :math:`P(\\beta) = \\lambda\\|\\beta\\|_1`.

    The proximal operator is soft-thresholding, which is well-defined
    for any positive step size.

    Examples
    --------
    >>> import numpy as np
    >>> from picpy.penalties import L1Penalty
    >>> pen = L1Penalty()
    >>> pen.evaluate(np.array([1.0, -2.0, 0.5]), lambda_=0.1)
    0.35
    """

    def evaluate(self, parameter: np.ndarray, lambda_: float) -> float:
        xp = get_namespace(parameter)
        return float(lambda_ * xp.sum(xp.abs(parameter)))

    def prox(
        self, parameter: np.ndarray, lambda_: float, step_size: float
    ) -> np.ndarray:
        return soft_thresholding(parameter, lambda_ * step_size)


class SCADPenalty(Penalty):
    """Smoothly Clipped Absolute Deviation (SCAD) penalty.

    Equals the L1 penalty near the origin and is constant beyond
    :math:`a\\lambda`, producing approximate unbiasedness for large
    coefficients.

    Parameters
    ----------
    a : float, default=3.7
        Concavity parameter (:math:`a > 2`).  The default value of 3.7
        is recommended by Fan and Li (2001).

    Notes
    -----
    The proximal operator's intermediate-region formula contains the
    factor :math:`(a-1)/(a-1-t)` where :math:`t` is the step size.
    This becomes singular at :math:`t = a-1` and changes sign for
    :math:`t > a-1`, so callers must enforce ``step_size < a - 1``.

    References
    ----------
    .. [1] Fan, J. and Li, R. (2001). Variable selection via nonconcave
       penalized likelihood and its oracle properties. *JASA*, 96(456),
       1348–1360.

    Examples
    --------
    >>> import numpy as np
    >>> from picpy.penalties import SCADPenalty
    >>> pen = SCADPenalty(a=3.7)
    >>> pen.evaluate(np.array([0.5, 0.0, -2.0]), lambda_=0.5)
    """

    def __init__(self, a: float = 3.7):
        if a <= 2.0:
            raise ValueError(f"SCAD requires a > 2, got a={a}")
        self.a = a

    def _param_str(self) -> str:
        return f"a={self.a}"

    @property
    def max_step(self) -> float:
        """Step-size upper bound for the SCAD prox: :math:`t < a - 1`."""
        return self.a - 1.0

    def evaluate(self, parameter: np.ndarray, lambda_: float) -> float:
        a = self.a
        xp = get_namespace(parameter)
        abs_param = xp.abs(parameter)

        m1 = abs_param <= lambda_
        m2 = (abs_param > lambda_) & (abs_param <= a * lambda_)

        p1 = lambda_ * abs_param
        p2 = (2.0 * a * lambda_ * abs_param - abs_param**2 - lambda_**2) / (
            2.0 * (a - 1.0)
        )
        p3 = (a + 1.0) * lambda_**2 / 2.0
        p = xp.where(m1, p1, xp.where(m2, p2, p3))

        return float(xp.sum(p))

    def prox(
        self, parameter: np.ndarray, lambda_: float, step_size: float
    ) -> np.ndarray:
        if step_size >= self.max_step:
            raise ValueError(
                f"SCAD prox requires step_size < a - 1 = {self.max_step}, "
                f"got step_size={step_size}. The solver should clip the "
                f"step using `penalty.max_step` before calling prox."
            )
        return scad_thresholding(parameter, lambda_, self.a, step_size)


class MCPPenalty(Penalty):
    """Minimax Concave Penalty (MCP).

    Reduces the L1 bias by linearly relaxing the penalty beyond
    :math:`\\gamma\\lambda`, yielding exact selection consistency under
    milder conditions than SCAD.

    Parameters
    ----------
    gamma : float, default=3.0
        Concavity parameter (:math:`\\gamma > 1`).

    Notes
    -----
    The proximal operator's soft-threshold-with-rescaling formula
    contains the factor :math:`\\gamma/(\\gamma - t)` where :math:`t` is
    the step size.  This becomes singular at :math:`t = \\gamma` and
    changes sign for :math:`t > \\gamma`, so callers must enforce
    ``step_size < gamma``.

    References
    ----------
    .. [1] Zhang, C.-H. (2010). Nearly unbiased variable selection under
       minimax concave penalty. *Annals of Statistics*, 38(2), 894–942.

    Examples
    --------
    >>> import numpy as np
    >>> from picpy.penalties import MCPPenalty
    >>> pen = MCPPenalty(gamma=3.0)
    >>> pen.evaluate(np.array([0.5, 0.0, -2.0]), lambda_=0.5)
    """

    def __init__(self, gamma: float = 3.0):
        if gamma <= 1.0:
            raise ValueError(f"MCP requires gamma > 1, got gamma={gamma}")
        self.gamma = gamma

    def _param_str(self) -> str:
        return f"gamma={self.gamma}"

    @property
    def max_step(self) -> float:
        """Step-size upper bound for the MCP prox: :math:`t < \\gamma`."""
        return self.gamma

    def evaluate(self, parameter: np.ndarray, lambda_: float) -> float:
        gamma = self.gamma
        xp = get_namespace(parameter)
        abs_param = xp.abs(parameter)

        m1 = abs_param <= lambda_ * gamma
        p1 = lambda_ * abs_param - abs_param**2 / (2.0 * gamma)
        p = xp.where(m1, p1, gamma * lambda_**2 / 2.0)

        return float(xp.sum(p))

    def prox(
        self, parameter: np.ndarray, lambda_: float, step_size: float
    ) -> np.ndarray:
        if step_size >= self.max_step:
            raise ValueError(
                f"MCP prox requires step_size < gamma = {self.max_step}, "
                f"got step_size={step_size}. The solver should clip the "
                f"step using `penalty.max_step` before calling prox."
            )
        return mcp_thresholding(parameter, lambda_, self.gamma, step_size)
