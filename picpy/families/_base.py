import numpy as np

from ..utils._array_api import get_namespace
from .links import Link, IdentityLink


class Family:
    """Abstract base class for a picpy families.

    A family bundles three things required by the PIC fitting algorithm:

    1. A **loss** function :math:`\\ell(y, \\hat\\mu)` and its gradient.
    2. A **mean-function link** *g* mapping the linear predictor
       :math:`\\eta = X\\beta + b` to the response scale: :math:`\\hat\\mu = g(\\eta)`.
    3. A **variance-stabilizing transform** *phi* applied to the raw loss
       before differentiation.

    The composite objective minimized by FISTA is therefore

    .. math::

        \\phi\\!\\left(\\ell(y,\\, g(X\\beta + b))\\right) + \\lambda\\, P(\\beta)

    Subclasses must implement :meth:`raw_loss` and :meth:`raw_loss_derivative`.

    Attributes
    ----------
    g : Link
        Mean-function link: maps linear predictor → fitted values.
    phi : Link
        Variance-stabilizing transform applied to the raw loss scalar.
    y_kind : {"continuous", "binary"}
        Type of response the family accepts; used to validate *y* in ``fit``.
    """

    g: Link = IdentityLink()
    phi: Link = IdentityLink()
    y_kind: str = "continuous"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(phi={self.phi}, g={self.g})"

    def __repr__(self) -> str:
        return self.__str__()

    def raw_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Unstabilized loss before applying *phi*.

        Parameters
        ----------
        y_true : ndarray of shape (n_samples,)
            Observed response(s).
        y_pred : ndarray of shape (n_samples,)
            Fitted values on the response scale (i.e. after applying *g*).

        Returns
        -------
        float or ndarray
            Scalar loss (or column-wise loss for batched null simulations).
        """
        raise NotImplementedError

    def raw_loss_derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Element-wise derivative of the raw loss w.r.t. *y_pred*.

        Parameters
        ----------
        y_true : ndarray
            Observed response.
        y_pred : ndarray
            Fitted values on the response scale.

        Returns
        -------
        ndarray of the same shape as *y_pred*
        """
        raise NotImplementedError

    def starting_intercept(self, y: np.ndarray) -> float:
        """Compute a reasonable initial intercept estimate.

        The default implementation inverts the mean response through *g*:
        :math:`b_0 = g^{-1}(\\bar y)`.

        Parameters
        ----------
        y : ndarray of shape (n_samples,) or (n_samples, n_simu)
            Response vector(s).

        Returns
        -------
        float or ndarray
        """
        xp = get_namespace(y)
        y = y if xp.isdtype(y.dtype, "real floating") else xp.astype(y, xp.float64)
        return self.g.inverse(xp.mean(y, axis=0))

    def evaluate(self, y: np.ndarray, lin_pred: np.ndarray) -> float:
        """Compute the (stabilized) loss at linear predictor *lin_pred*.

        Applies the mean-function link *g* to obtain fitted values, evaluates
        :meth:`raw_loss`, and then applies *phi*:

        .. math::

            \\phi\\!\\bigl(\\ell(y,\\; g(\\eta))\\bigr)

        Parameters
        ----------
        y : ndarray
            Observed response.
        lin_pred : ndarray
            Linear predictor :math:`\\eta = X\\beta + b`.

        Returns
        -------
        float
            Scalar stabilized loss.
        """
        y_pred = self.g(lin_pred)
        raw = self.raw_loss(y_true=y, y_pred=y_pred)
        return self.phi(raw)

    def grad(
        self, X: np.ndarray, y: np.ndarray, lin_pred: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Gradient of the stabilized loss w.r.t. coefficients and intercept.

        Uses the chain rule through *phi*, *raw_loss*, and *g*:

        .. math::

            \\nabla_{\\beta}\\, \\phi(\\ell) =
            \\phi'(\\ell)\\cdot X^{\\top}
            \\left[g'(\\eta) \\odot \\ell'(y, g(\\eta))\\right]

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Design matrix.
        y : ndarray
            Observed response.
        lin_pred : ndarray of shape (n_samples,)
            Linear predictor.

        Returns
        -------
        coef_grad : ndarray of shape (n_features,)
            Gradient w.r.t. regression coefficients.
        intercept_grad : float or ndarray of shape (n_simu,)
            Gradient w.r.t. the intercept.
        """
        y_pred = self.g(lin_pred)
        raw = self.raw_loss(y_true=y, y_pred=y_pred)
        phi_prime = self.phi.derivative(raw)
        g_prime = self.g.derivative(lin_pred)
        residual = (
            phi_prime * g_prime * self.raw_loss_derivative(y_true=y, y_pred=y_pred)
        )
        coef_grad = X.T @ residual
        intercept_grad = get_namespace(residual).sum(residual, axis=0)
        return coef_grad, intercept_grad
