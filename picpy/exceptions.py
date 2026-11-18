"""Custom warnings raised by picpy."""


class ConvergenceWarning(UserWarning):
    """Raised when the solver stops before reaching the requested tolerance."""
