"""
Non-monotone Accelerated Proximal Gradient (nmAPG) with line search.

Faithful implementation of Algorithm 4 of
    H. Li and Z. Lin, "Accelerated Proximal Gradient Methods for Nonconvex
    Programming", NeurIPS 2015 (Supplementary Material).
"""

import warnings
import numpy as np
from typing import Tuple, Optional

from array_api_compat import device

from ..families import Family
from ..penalties import Penalty
from ..utils import TrainingLogger
from ..utils._array_api import get_namespace, to_numpy

# Safety margin applied to penalty.max_step.  Strictly below 1 because the
# non-convex prox formulas (SCAD, MCP) blow up as the step approaches the
# bound; 0.95 keeps us well clear of the singularity while leaving plenty
# of room for the solver.
_STEP_MARGIN = 0.95


def fista(
    X: np.ndarray,
    y: np.ndarray,
    family: Family,
    penalty: Penalty,
    lambda_reg: float,
    fit_intercept: bool = True,
    rel_tol: float = 1e-4,
    step_size_init: float = 1e-1,
    max_iter: int = 500,
    eta: float = 0.8,
    delta: float = 1e-4,
    rho: float = 0.5,
    bb_growth_cap: float = 4.0,
    logger: Optional[TrainingLogger] = None,
    coef_init: Optional[np.ndarray] = None,
    intercept_init: Optional[float] = None,
) -> Tuple[np.ndarray, Optional[float], dict]:
    """
    Non-monotone Accelerated Proximal Gradient with line search.

    Minimizes :math:`F(\\beta) = f(\\beta) + g(\\beta)` where *f* is the
    smooth family loss and *g* the (possibly non-convex) penalty.

    Each outer iteration *k*:

    1. Build the extrapolated point
       :math:`y_k = x_k + (t_{k-1}/t_k)(z_k - x_k)
       + ((t_{k-1}-1)/t_k)(x_k - x_{k-1})`.
    2. Compute a Barzilai–Borwein step :math:`\\alpha_y` on
       :math:`(s_k, r_k) = (y_k - y_{k-1},\\ \\nabla f(y_k) - \\nabla f(y_{k-1}))`,
       capped relative to the previous step and to ``penalty.max_step``.
    3. Proximal line search from :math:`y_k`: repeatedly shrink
       :math:`\\alpha_y \\leftarrow \\rho\\alpha_y` and recompute
       :math:`z_{k+1} = \\mathrm{prox}_{\\alpha_y g}(y_k - \\alpha_y\\nabla f(y_k))`
       until the descent lemma on *f* holds.
    4. If :math:`F(z_{k+1}) \\le c_k - \\delta\\|z_{k+1}-y_k\\|^2`, accept
       :math:`x_{k+1} = z_{k+1}`.  Otherwise, BB on
       :math:`(x_k - y_{k-1}, \\nabla f(x_k) - \\nabla f(y_{k-1}))` and run
       a proximal line search from :math:`x_k`; keep whichever of
       :math:`z_{k+1}, v_{k+1}` has lower :math:`F`.
    5. Update :math:`t_{k+1} = (\\sqrt{4t_k^2+1}+1)/2`,
       :math:`q_{k+1} = \\eta q_k + 1`, and the non-monotone reference
       :math:`c_{k+1} = (\\eta q_k c_k + F(x_{k+1}))/q_{k+1}`.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Design matrix (assumed standardized).
    y : ndarray of shape (n_samples,) or (n_samples, 2)
        Response vector or survival matrix.
    family : Family
        Loss with ``evaluate(y, lin_pred)`` and
        ``grad(X, y, lin_pred) -> (coef_grad, intercept_grad)``.
    penalty : Penalty
        Regularization penalty providing ``evaluate``, ``prox`` and the
        ``max_step`` property (strict upper bound on admissible step).
    lambda_reg : float
        Regularization parameter :math:`\\lambda`.
    fit_intercept : bool, default=True
        Update an unpenalized intercept jointly with the coefficients.
    rel_tol : float, default=1e-4
        Convergence tolerance on the relative gradient-mapping norm
        :math:`\\|G_t(\\beta)\\| / \\max(\\|\\beta\\|, 1) < \\varepsilon`, where
        :math:`G_t(\\beta) = (\\beta - \\mathrm{prox}_{t\\lambda P}(\\beta - t\\nabla q(\\beta)))/t`.
        The anchor :math:`\\beta` and step :math:`t` match whichever
        proximal step produced :math:`x_{k+1}`: when :math:`x_{k+1}=z_{k+1}`,
        :math:`(\\beta, t) = (y_k, \\alpha_y)` and
        :math:`G = (y_k - z_{k+1})/\\alpha_y`; when :math:`x_{k+1}=v_{k+1}`,
        :math:`(\\beta, t) = (x_k, \\alpha_x)` and
        :math:`G = (x_k - v_{k+1})/\\alpha_x`. In both cases the value is
        already computed by the proximal line search.  When
        ``fit_intercept`` the intercept is included in both norms.  The
        ``max(., 1)`` makes the test absolute near :math:`\\beta = 0`.
    step_size_init : float, default=1e-1
        Initial step size used until BB has two iterates to compare.
    max_iter : int, default=500
        Safety cap on outer iterations.
    eta : float, default=0.8
        :math:`\\eta \\in [0, 1)`; memory weight of the non-monotone
        reference :math:`c_k`.  ``eta=0`` recovers a strictly monotone
        scheme; values near 1 average over many past iterates.
    delta : float, default=1e-4
        Sufficient-descent constant in the line-search conditions.
    rho : float, default=0.5
        Step-size reduction factor in the line search (:math:`\\rho < 1`).
    bb_growth_cap : float, default=4.0
        Maximum factor by which the BB step is allowed to grow over the
        previous accepted step.  Prevents catastrophic jumps that, with
        non-convex penalties, can land the iterate in a bad local
        minimum.
    logger : TrainingLogger or None
        Optional logger for iteration-level diagnostics.
    coef_init, intercept_init
        Optional warm starts; default to zeros and the family-specific
        starting intercept respectively.

    Returns
    -------
    coef : ndarray of shape (n_features,)
    intercept : float or None
    info : dict
        Keys: ``'n_iter'``, ``'lambda'``, ``'objective_path'``,
        ``'features_path'``, ``'status'`` (``'converged'`` or
        ``'max_iter_reached'``), ``'linesearch_warnings'``.
    """
    n_features = X.shape[1]
    xp = get_namespace(X)
    detailed_log = logger is not None and logger.verbosity >= logger.DETAILED

    # Hard upper bound on every step we ever pass to penalty.prox.
    alpha_pen_cap = _STEP_MARGIN * penalty.max_step

    # Sanity-clip the user-supplied initial step too, otherwise the very
    # first prox call could violate the penalty's domain.
    step_size_init = min(step_size_init, alpha_pen_cap)

    # =========== Initialize: z_1 = x_1 = x_0, t_1 = 1, t_0 = 0 ===========
    if coef_init is None:
        x_coef = xp.zeros(n_features, dtype=X.dtype, device=device(X))
    else:
        x_coef = xp.asarray(coef_init, dtype=X.dtype, device=device(X), copy=True)
    if fit_intercept:
        x_int = float(
            family.starting_intercept(y)
            if intercept_init is None
            else intercept_init
        )
    else:
        x_int = 0.0

    x_prev_coef, x_prev_int = _copy(xp, x_coef), x_int
    z_coef, z_int = _copy(xp, x_coef), x_int

    # y_0 := x_0 so that at k=1 the BB difference s_1 = y_1 - y_0 = 0,
    # falling back cleanly to step_size_init.
    y_prev_coef, y_prev_int = _copy(xp, x_coef), x_int
    lp = X @ x_coef + x_int
    grad_y_prev_coef, grad_y_prev_int = _grad(family, X, y, lp)

    t_prev, t_curr = 0.0, 1.0

    # c_1 = F(x_1), q_1 = 1
    F_x = float(family.evaluate(y, lp) + penalty.evaluate(x_coef, lambda_reg))
    c_k, q_k = F_x, 1.0

    alpha_y = step_size_init
    alpha_x = step_size_init

    objective_path: list = []
    features_path: list = []
    n_ls_warnings = 0
    status = "max_iter_reached"
    n_iter = 0

    for k in range(max_iter):
        # =========== (1) Extrapolation ===========
        a = t_prev / t_curr
        b = (t_prev - 1.0) / t_curr
        y_coef = x_coef + a * (z_coef - x_coef) + b * (x_coef - x_prev_coef)
        if fit_intercept:
            y_int = x_int + a * (z_int - x_int) + b * (x_int - x_prev_int)
        else:
            y_int = 0.0

        lp_y = X @ y_coef + y_int
        f_y = float(family.evaluate(y, lp_y))
        F_y = f_y + float(penalty.evaluate(y_coef, lambda_reg))
        grad_y_coef, grad_y_int = _grad(family, X, y, lp_y)

        # =========== (2) BB step from (y_k, y_{k-1}) ===========
        alpha_y = _bb_step(
            y_coef,
            y_prev_coef,
            y_int,
            y_prev_int,
            grad_y_coef,
            grad_y_prev_coef,
            grad_y_int,
            grad_y_prev_int,
            fallback=alpha_y,
            fit_intercept=fit_intercept,
            growth_cap=bb_growth_cap,
            alpha_pen_cap=alpha_pen_cap,
        )

        # =========== (3) Proximal line search from y_k ===========
        z_next_coef, z_next_int, F_z, alpha_y, status_y = _ls_from_y(
            X,
            y,
            family,
            penalty,
            lambda_reg,
            y_coef,
            y_int,
            f_y,
            F_y,
            grad_y_coef,
            grad_y_int,
            min(alpha_y, alpha_pen_cap),
            c_k,
            delta,
            rho,
            fit_intercept,
        )

        if status_y == "failed":
            warnings.warn(
                f"nmAPG: line search from y failed at iteration {k}; "
                f"accepting last trial.",
                RuntimeWarning,
                stacklevel=2,
            )
            n_ls_warnings += 1

        # =========== (4) Decide x_{k+1} ===========
        # Track the (anchor, step) pair that produced x_next so that the
        # convergence test below uses the matching gradient mapping.
        if status_y == "nonmonotone":
            # Condition (b) reached → accept z directly, skip v-branch.
            x_next_coef = z_next_coef
            x_next_int = z_next_int
            F_next = F_z
            gm_anchor_coef, gm_anchor_int, gm_alpha = y_coef, y_int, alpha_y
        else:
            # Only (a) (or failed) → fall back to v-branch from x_k.
            x_next_coef, x_next_int, F_next, alpha_x, ls_w, picked = _v_branch(
                X,
                y,
                family,
                penalty,
                lambda_reg,
                x_coef,
                x_int,
                y_prev_coef,
                y_prev_int,
                grad_y_prev_coef,
                grad_y_prev_int,
                z_next_coef,
                z_next_int,
                F_z,
                alpha_x,
                c_k,
                delta,
                rho,
                fit_intercept,
                k,
                bb_growth_cap=bb_growth_cap,
                alpha_pen_cap=alpha_pen_cap,
            )
            n_ls_warnings += int(ls_w)
            if picked == "z":
                gm_anchor_coef, gm_anchor_int, gm_alpha = y_coef, y_int, alpha_y
            else:
                gm_anchor_coef, gm_anchor_int, gm_alpha = x_coef, x_int, alpha_x

        # =========== Logging ===========
        # Skipped below verbosity=2: the extra penalty evaluation and the
        # active-set count would each force a device synchronization.
        if detailed_log:
            smooth_loss = F_next - penalty.evaluate(x_next_coef, lambda_reg)
            logger.log_iteration(
                iteration=k,
                objective=F_next,
                loss=smooth_loss,
                n_active=int(xp.sum(x_next_coef != 0)),
            )

        objective_path.append(F_next)
        features_path.append(xp.nonzero(x_next_coef)[0])
        n_iter = k + 1

        # =========== Convergence: relative gradient-mapping norm ===========
        # Evaluated at the anchor that produced x_next:
        #   - z accepted: anchor = y_k,  step = alpha_y  → G = (y - z)/alpha_y
        #   - v accepted: anchor = x_k,  step = alpha_x  → G = (x - v)/alpha_x
        # Both are already computed by the line search — no extra grad/prox.
        G_coef = (gm_anchor_coef - x_next_coef) / gm_alpha
        G_sq = float(G_coef @ G_coef)
        anchor_sq = float(gm_anchor_coef @ gm_anchor_coef)
        if fit_intercept:
            G_int = (gm_anchor_int - x_next_int) / gm_alpha
            G_sq += G_int * G_int
            anchor_sq += gm_anchor_int * gm_anchor_int
        rel_gm = np.sqrt(G_sq) / max(np.sqrt(anchor_sq), 1.0)
        if rel_gm < rel_tol:
            x_coef = x_next_coef
            x_int = x_next_int
            status = "converged"
            if logger is not None:
                logger.log_convergence(n_iter=n_iter)
            break

        # =========== (5) State roll-forward ===========
        x_prev_coef = _copy(xp, x_coef)
        x_prev_int = x_int
        z_coef = _copy(xp, z_next_coef)
        z_int = z_next_int
        y_prev_coef = y_coef  # already a fresh array
        y_prev_int = y_int
        grad_y_prev_coef = grad_y_coef
        grad_y_prev_int = grad_y_int
        x_coef = x_next_coef
        x_int = x_next_int

        t_next = (np.sqrt(4 * t_curr**2 + 1) + 1) / 2
        t_prev, t_curr = t_curr, t_next

        q_next = eta * q_k + 1.0
        c_k = (eta * q_k * c_k + F_next) / q_next
        q_k = q_next

    info = {
        "n_iter": n_iter,
        "lambda": lambda_reg,
        "objective_path": objective_path,
        # Active sets are returned as NumPy index arrays whatever the backend.
        "features_path": [to_numpy(idx) for idx in features_path],
        "status": status,
        "linesearch_warnings": n_ls_warnings,
    }
    intercept_out = float(x_int) if fit_intercept else None
    return x_coef, intercept_out, info


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _copy(xp, a):
    """Backend-agnostic copy of a coefficient vector."""
    return xp.asarray(a, copy=True)


def _grad(family, X, y, lin_pred):
    """Family gradient with the intercept part returned as a Python float.

    Keeping intercepts as Python scalars avoids mixing 0-d device tensors
    with the solver's scalar bookkeeping.
    """
    coef_grad, intercept_grad = family.grad(X, y, lin_pred)
    return coef_grad, float(intercept_grad)


def _bb_step(
    coef_a,
    coef_b,
    int_a,
    int_b,
    grad_a_coef,
    grad_b_coef,
    grad_a_int,
    grad_b_int,
    fallback,
    fit_intercept,
    growth_cap,
    alpha_pen_cap,
    alpha_min=1e-10,
    alpha_max=1e10,
    eps=1e-12,
):
    """Barzilai–Borwein step :math:`\\alpha = (s^T s)/(s^T r)` on the joint
    (coef, intercept) vector.

    Returns ``fallback`` when the formula is not numerically well-defined
    (zero displacement or non-positive curvature).  The result is clipped
    to

        [alpha_min, min(alpha_max, growth_cap * fallback, alpha_pen_cap)]

    so that (i) the step cannot grow by more than ``growth_cap`` over the
    previous accepted step, and (ii) the step never enters the region
    where the penalty's prox is ill-defined.
    """
    s_coef = coef_a - coef_b
    r_coef = grad_a_coef - grad_b_coef
    ss = float(s_coef @ s_coef)
    sr = float(s_coef @ r_coef)
    if fit_intercept:
        s_int = int_a - int_b
        r_int = grad_a_int - grad_b_int
        ss += s_int * s_int
        sr += s_int * r_int
    if ss <= eps or sr <= eps:
        # BB undefined → keep previous step but still respect the cap.
        return float(min(fallback, alpha_pen_cap))

    upper = min(alpha_max, growth_cap * fallback, alpha_pen_cap)
    return float(np.clip(ss / sr, alpha_min, upper))


def _ls_from_y(
    X,
    y,
    family,
    penalty,
    lambda_reg,
    y_coef,
    y_int,
    f_y,
    F_y,
    grad_y_coef,
    grad_y_int,
    alpha,
    c_k,
    delta,
    rho,
    fit_intercept,
    max_inner=30,
):
    """Proximal line search from y_k under the descent-lemma condition.

    Repeats ``z = prox(y - alpha grad)`` then ``alpha *= rho`` until the
    quadratic upper bound on the smooth part holds:

        f(z) <= f(y) + <grad f(y), z - y> + ||z - y||^2 / (2 alpha).

    The non-monotone F-comparison is then evaluated *after* the line
    search to choose the branch:

        F(z) <= c_k - delta * ||z - y||^2  → 'nonmonotone' (skip v-branch),
        otherwise                           → 'sufficient' (go to v-branch).

    Returns 'failed' when the descent lemma did not hold within
    ``max_inner`` trials.

    The caller is responsible for ensuring that ``alpha`` is below
    ``penalty.max_step`` on entry; subsequent shrinks by ``rho`` only
    decrease it, so no further capping is needed inside the loop.
    """
    z_coef, z_int, F_z = y_coef, y_int, F_y
    f_z = f_y
    diff_sq = 0.0

    for _ in range(max_inner):
        z_coef = penalty.prox(y_coef - alpha * grad_y_coef, lambda_reg, alpha)
        if fit_intercept:
            z_int = y_int - alpha * grad_y_int
        else:
            z_int = 0.0

        lp = X @ z_coef + z_int
        f_z = float(family.evaluate(y, lp))

        d_coef = z_coef - y_coef
        inner = float(d_coef @ grad_y_coef)
        diff_sq = float(d_coef @ d_coef)
        if fit_intercept:
            d_int = z_int - y_int
            inner += d_int * grad_y_int
            diff_sq += d_int * d_int

        # Descent lemma on f: f(z) <= f(y) + <grad, z-y> + ||z-y||^2/(2 alpha)
        if f_z <= f_y + inner + diff_sq / (2.0 * alpha):
            F_z = f_z + float(penalty.evaluate(z_coef, lambda_reg))
            if F_z <= c_k - delta * diff_sq:
                return z_coef, z_int, F_z, alpha, "nonmonotone"
            return z_coef, z_int, F_z, alpha, "sufficient"

        alpha *= rho

    F_z = f_z + float(penalty.evaluate(z_coef, lambda_reg))
    return z_coef, z_int, F_z, alpha, "failed"


def _v_branch(
    X,
    y,
    family,
    penalty,
    lambda_reg,
    x_coef,
    x_int,
    y_prev_coef,
    y_prev_int,
    grad_y_prev_coef,
    grad_y_prev_int,
    z_next_coef,
    z_next_int,
    F_z,
    alpha_x,
    c_k,
    delta,
    rho,
    fit_intercept,
    iteration,
    bb_growth_cap,
    alpha_pen_cap,
    max_inner=30,
):
    """Else branch of Algorithm 4.

    BB step on :math:`(s, r) = (x_k - y_{k-1},\\ \\nabla f(x_k) - \\nabla f(y_{k-1}))`,
    then proximal line search from x_k under the descent-lemma condition

        f(v) <= f(x) + <grad f(x), v - x> + ||v - x||^2 / (2 alpha_x).

    Returns whichever of z and v has lower F, plus a tag ``"z"`` or
    ``"v"`` indicating which iterate was selected (so the caller can pick
    the matching gradient mapping for its convergence test).
    """
    lp_x = X @ x_coef + x_int
    f_x = float(family.evaluate(y, lp_x))
    grad_x_coef, grad_x_int = _grad(family, X, y, lp_x)

    alpha_x = _bb_step(
        x_coef,
        y_prev_coef,
        x_int,
        y_prev_int,
        grad_x_coef,
        grad_y_prev_coef,
        grad_x_int,
        grad_y_prev_int,
        fallback=alpha_x,
        fit_intercept=fit_intercept,
        growth_cap=bb_growth_cap,
        alpha_pen_cap=alpha_pen_cap,
    )
    alpha_x = min(alpha_x, alpha_pen_cap)

    v_coef, v_int = x_coef, x_int
    f_v = f_x
    F_v = np.inf
    ls_failed = True
    for _ in range(max_inner):
        v_coef = penalty.prox(x_coef - alpha_x * grad_x_coef, lambda_reg, alpha_x)
        if fit_intercept:
            v_int = x_int - alpha_x * grad_x_int
        else:
            v_int = 0.0

        lp = X @ v_coef + v_int
        f_v = float(family.evaluate(y, lp))

        d_coef = v_coef - x_coef
        inner = float(d_coef @ grad_x_coef)
        diff_sq = float(d_coef @ d_coef)
        if fit_intercept:
            d_int = v_int - x_int
            inner += d_int * grad_x_int
            diff_sq += d_int * d_int

        # Descent lemma on f.
        if f_v <= f_x + inner + diff_sq / (2.0 * alpha_x):
            F_v = f_v + float(penalty.evaluate(v_coef, lambda_reg))
            ls_failed = False
            break
        alpha_x *= rho

    if ls_failed:
        F_v = f_v + float(penalty.evaluate(v_coef, lambda_reg))
        warnings.warn(
            f"nmAPG: line search from x failed at iteration {iteration}; "
            f"accepting last trial.",
            RuntimeWarning,
            stacklevel=3,
        )

    if F_z <= F_v:
        return z_next_coef, z_next_int, F_z, alpha_x, ls_failed, "z"
    return v_coef, v_int, F_v, alpha_x, ls_failed, "v"
