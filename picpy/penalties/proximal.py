"""Proximal operators for sparsity-inducing penalties.

These functions implement the closed-form proximal maps used inside the FISTA
solver for the L1, SCAD, and MCP penalties.  They are written with
element-wise ``where`` selections so that they run unchanged on NumPy arrays
and on PyTorch tensors (CPU or GPU).
"""

from ..utils._array_api import get_namespace


def soft_thresholding(parameter, threshold: float):
    """Soft-thresholding operator (proximal map of the L1 penalty).

    .. math::

        \\operatorname{prox}_{t\\lambda\\|\\cdot\\|_1}(u) =
        \\operatorname{sign}(u)\\,\\max(|u| - t\\lambda,\\; 0)

    Parameters
    ----------
    parameter : ndarray or torch.Tensor
        Input vector :math:`u`.
    threshold : float
        Combined regularization-step product :math:`t\\lambda`.

    Returns
    -------
    ndarray or torch.Tensor
        Thresholded vector.
    """
    xp = get_namespace(parameter)
    return xp.sign(parameter) * xp.maximum(xp.abs(parameter) - threshold, 0.0)


def mcp_thresholding(
    parameter,
    lambda_: float,
    gamma: float,
    step_size: float,
):
    """Proximal map of the Minimax Concave Penalty (MCP).

    Applies soft-thresholding on the region where the MCP is active and
    leaves the parameter unchanged in the unpenalized flat region.

    Parameters
    ----------
    parameter : ndarray or torch.Tensor
        Input vector.
    lambda_ : float
        Regularization parameter.
    gamma : float
        MCP concavity parameter (:math:`\\gamma > 1`).
    step_size : float
        FISTA step size (backtracking parameter).

    Returns
    -------
    ndarray or torch.Tensor
    """
    xp = get_namespace(parameter)
    mid = xp.abs(parameter) <= lambda_ * gamma
    shrunk = (gamma / (gamma - step_size)) * soft_thresholding(
        parameter, lambda_ * step_size
    )
    return xp.where(mid, shrunk, parameter)


def scad_thresholding(
    parameter,
    lambda_: float,
    a: float,
    step_size: float,
):
    """Proximal map of the Smoothly Clipped Absolute Deviation (SCAD) penalty.

    Applies a piecewise proximal rule with three regions corresponding to
    the three pieces of the SCAD penalty.

    Parameters
    ----------
    parameter : ndarray or torch.Tensor
        Input vector.
    lambda_ : float
        Regularization parameter.
    a : float
        SCAD concavity parameter (:math:`a > 2`, typically :math:`a = 3.7`).
    step_size : float
        FISTA step size (backtracking parameter).

    Returns
    -------
    ndarray or torch.Tensor
    """
    xp = get_namespace(parameter)
    abs_param = xp.abs(parameter)

    m1 = abs_param <= lambda_ * (1.0 + step_size)
    m2 = (abs_param > lambda_ * (1.0 + step_size)) & (abs_param <= a * lambda_)

    region1 = soft_thresholding(parameter, lambda_ * step_size)
    region2 = ((a - 1.0) / (a - 1.0 - step_size)) * soft_thresholding(
        parameter, (a * lambda_ * step_size) / (a - 1.0)
    )
    return xp.where(m1, region1, xp.where(m2, region2, parameter))
