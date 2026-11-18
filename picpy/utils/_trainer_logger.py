import numpy as np
from typing import Optional


class TrainingLogger:
    """Structured progress logger for model fitting.

    Prints formatted, consistent output at three verbosity levels.  All
    output goes to stdout via :func:`print`.

    Parameters
    ----------
    verbosity : int, default=0
        Controls the amount of output:

        - ``0`` (silent)   — no output.
        - ``1`` (basic)    — high-level progress: per-step
          convergence summary and final results.
        - ``2`` (detailed) — everything in level 1, plus per-iteration
          objective values inside each FISTA run.

    log_interval : int, default=10
        At ``verbosity=2``, one iteration line is printed every
        *log_interval* iterations.  Has no effect at lower levels.
    """

    SILENT = 0
    BASIC = 1
    DETAILED = 2

    # Column widths and indentation
    _INDENT = " " * 8  # continuation lines aligned with tag content
    _STEP_INDENT = " " * 10  # iteration/convergence lines nested in a step
    _TAG_WIDTH = 8  # total width of "[TAG]   " prefix field

    def __init__(self, verbosity: int = 0, log_interval: int = 10):
        if verbosity not in (self.SILENT, self.BASIC, self.DETAILED):
            raise ValueError(f"verbosity must be 0, 1, or 2; got {verbosity!r}.")
        self.verbosity = verbosity
        self.log_interval = log_interval

        # Internal state
        self._step_cache: Optional[tuple] = None  # (idx, total, lambda, tol)
        self._in_refit: bool = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _emit(self, line: str, min_level: int = 1) -> None:
        """Print *line* when ``self.verbosity >= min_level``."""
        if self.verbosity >= min_level:
            print(line)

    def _tag(self, tag: str, content: str) -> str:
        """Format a tagged section header: ``'[TAG]   content'``."""
        return f"[{tag}]".ljust(self._TAG_WIDTH) + content

    def _cont(self, content: str) -> str:
        """Format a continuation line indented to match tag content."""
        return self._INDENT + content

    def _step(self, content: str) -> str:
        """Format a line nested inside a path step (deeper indent)."""
        return self._STEP_INDENT + content

    def _format_step_header(self, n_iter: Optional[int]) -> str:
        """Build the per-step header line from the cached metadata.

        The trailing ``converged (N iter)`` segment is appended only when
        *n_iter* is supplied (i.e. at convergence time).
        """
        if self._step_cache is None:
            return ""
        idx, total, lam, tol, phase, penalty_name = self._step_cache
        # Pad phase to width 4 so "Path"/"LLA " line up across the log.
        line = f"{phase:<4} {idx + 1}/{total}" f"  lambda={lam:.6f}" f"  tol={tol:.1e}"
        if penalty_name:
            # Pad penalty name to a fixed width to keep columns aligned.
            line += f"  penalty={penalty_name:<22}"
        if n_iter is not None:
            line += f"  converged ({n_iter} iter)"
        return line

    # ------------------------------------------------------------------
    # Regularization path
    # ------------------------------------------------------------------

    def start_fitting(
        self,
        n_samples: int,
        n_features: int,
        lambda_reg: float,
        penalty_name: str,
    ) -> None:
        """Log the start of the regularization path.

        Parameters
        ----------
        n_samples : int
            Number of observations.
        n_features : int
            Number of predictors.
        lambda_reg : float
            Target regularization parameter (lambda PDB).
        penalty_name : str
            String representation of the penalty.
        """
        self._emit("")
        self._emit(
            self._tag(
                "Fit",
                f"n={n_samples:,}  p={n_features:,}"
                f"  lambda={lambda_reg:.6f}  penalty={penalty_name}",
            )
        )

    def log_path_start(
        self,
        path_idx: int,
        total_paths: int,
        lambda_current: float,
        rel_tol: float,
        phase: str = "Path",
        penalty_name: Optional[str] = None,
    ) -> None:
        """Log the start of one step in the regularization path.

        At ``verbosity=2`` the step header is printed immediately so that
        subsequent iteration lines appear below it.  At ``verbosity=1`` the
        header is held in cache and emitted together with the convergence
        result once :meth:`log_convergence` is called.

        Parameters
        ----------
        path_idx : int
            Zero-based step index within *phase*.
        total_paths : int
            Total number of steps within *phase*.
        lambda_current : float
            Regularization parameter for this step.
        rel_tol : float
            Relative convergence tolerance passed to FISTA.
        phase : str, default="Path"
            Name of the fitting phase, used as the step label.  Typical
            values are ``"Path"`` for the warm-started regularization path
            and ``"LLA"`` for the outer iterations of the Local Linear
            Approximation solver.
        penalty_name : str or None, default=None
            String representation of the penalty actually used at this step
            (e.g. ``"L1Penalty"`` or ``"WeightedL1Penalty"``).  Included in
            the output line when provided.
        """
        self._step_cache = (
            path_idx,
            total_paths,
            lambda_current,
            rel_tol,
            phase,
            penalty_name,
        )
        self._in_refit = False

        if self.verbosity >= self.DETAILED:
            self._emit(self._cont(self._format_step_header(n_iter=None)))

    def log_iteration(
        self,
        iteration: int,
        objective: float,
        loss: Optional[float] = None,
        n_active: Optional[int] = None,
    ) -> None:
        """Log one FISTA iteration.

        Only printed when ``verbosity=2`` and ``(iteration + 1)`` is a
        multiple of ``log_interval``.

        Parameters
        ----------
        iteration : int
            Zero-based iteration index.
        objective : float
            Current penalized objective value.
        loss : float or None
            Current unpenalized (smooth) loss.  Omitted during the
            unpenalized refit phase.
        n_active : int or None
            Number of non-zero coefficients.  Omitted during refit.
        """
        if self.verbosity < self.DETAILED:
            return
        if (iteration + 1) % self.log_interval != 0:
            return

        parts = [f"Iter {iteration + 1:5d} | obj={objective:.6f}"]
        if loss is not None:
            parts.append(f"loss={loss:.6f}")
        if n_active is not None and not self._in_refit:
            parts.append(f"active={n_active:3d}")

        self._emit(self._step("  ".join(parts)), min_level=self.DETAILED)

    def log_convergence(self, n_iter: int) -> None:
        """Log convergence of a FISTA run.

        At ``verbosity=1``, prints a compact one-line step summary using
        the metadata cached by the most recent :meth:`log_path_start` or
        :meth:`start_refit` call.  At ``verbosity=2``, prints an indented
        convergence notice following any iteration lines.

        Parameters
        ----------
        n_iter : int
            Total number of FISTA iterations performed.
        """
        if self.verbosity == self.BASIC:
            if self._in_refit:
                self._emit(
                    self._cont(f"Refit (unpenalized)  converged ({n_iter} iter)")
                )
            elif self._step_cache is not None:
                self._emit(self._cont(self._format_step_header(n_iter=n_iter)))
        elif self.verbosity >= self.DETAILED:
            self._emit(self._step(f"converged in {n_iter} iterations"))

    # ------------------------------------------------------------------
    # Refit phase
    # ------------------------------------------------------------------

    def start_refit(self, n_selected: int) -> None:
        """Log the start of the unpenalized refit on the selected support.

        Parameters
        ----------
        n_selected : int
            Number of features in the selected support.
        """
        self._in_refit = True
        if self.verbosity >= self.DETAILED:
            self._emit(
                self._cont(f"Refit (unpenalized)  support={n_selected} features")
            )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------

    def log_final_results(
        self,
        n_selected: int,
        objective_final: float,
        extra_metrics: Optional[dict] = None,
    ) -> None:
        """Log the final model summary after fitting completes.

        Parameters
        ----------
        n_selected : int
            Number of selected features (non-zero coefficients).
        objective_final : float
            Final penalized objective value.
        extra_metrics : dict or None
            Optional additional metrics to display (e.g. training score).
            Keys are metric names; values are floats.
        """
        self._emit("")
        summary = f"selected={n_selected}   objective={objective_final:.6f}"
        self._emit(self._tag("Done", summary))
        if extra_metrics:
            for name, value in extra_metrics.items():
                self._emit(self._cont(f"{name} = {value:.6f}"))
