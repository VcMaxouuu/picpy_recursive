import numpy as np

from ._base import _BasePicModel
from ..families import Gaussian
from ..penalties import Penalty, L1Penalty
from ..utils._array_api import to_numpy


class LinearRegression(_BasePicModel):
    """Penalized regression for continuous responses.

    Parameters
    ----------
    lambda_ : float
        Regularization parameter (> 0). Should be computed by a
        `GaussianCalibrator` on the same standardized *X* that is passed to
        :meth:`fit`.
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
        Amount of output printed during :meth:`fit` (0, 1 or 2).
    log_interval : int, default=5
        At ``verbosity=2``, print one iteration line every *log_interval*
        iterations.
    device : None, str or torch.device, default=None
        Compute device. ``None`` follows the input (NumPy on CPU, tensors on
        their device); ``"cuda"`` moves the data to the GPU. ``coef_`` is a
        NumPy array for NumPy inputs and a tensor for tensor inputs.
    dtype : {"float64", "float32"}, default="float64"
        Solver precision. ``"float64"`` reproduces the NumPy results;
        ``"float32"`` is faster and lighter on a GPU but may change the
        supports selected by SCAD/MCP.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Fitted regression coefficients.
    intercept_ : float or None
        Fitted intercept.  None when *fit_intercept=False*.
    loss_ : float
        Smooth (stabilized) loss at the fitted coefficients.
    objective_ : float
        Penalized objective at the fitted coefficients.
    n_iter_ : int
        Number of FISTA iterations of the last solver run.
    converged_ : bool
        Whether the final solver run(s) reached *tol*.
    selected_features_ : ndarray of int
        Indices of features with non-zero fitted coefficients.
    is_fitted_ : bool
        True after :meth:`fit` has been called.
    """

    def __init__(
        self,
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
        super().__init__(
            family=Gaussian(),
            lambda_=lambda_,
            penalty=penalty,
            fit_intercept=fit_intercept,
            refit=refit,
            tol=tol,
            max_iter=max_iter,
            verbosity=verbosity,
            log_interval=log_interval,
            device=device,
            dtype=dtype,
        )

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute the coefficient of determination :math:`R^2` on test data.

        .. math::

            R^2 = 1 - \\frac{\\sum_i (y_i - \\hat y_i)^2}
                            {\\sum_i (y_i - \\bar y)^2}

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test design matrix.
        y : array-like of shape (n_samples,)
            True response values.

        Returns
        -------
        float
            :math:`R^2` score.  1.0 indicates a perfect fit; 0.0 is the
            score of a constant (mean) predictor.
        """
        y = to_numpy(y).astype(float)
        y_pred = to_numpy(self.predict(X))
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
