"""
Logger protocol and base implementations.
"""
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from foresight_metrics.results import MetricResult


class MetricsLogger(Protocol):
    """
    Protocol for metric loggers.

    Implement the `log` method to create a custom logger that sends
    metrics to any destination (terminal, file, experiment tracker, etc.).

    Example:
        >>> class MyCustomLogger:
        ...     def log(self, result: MetricResult) -> None:
        ...         # Send metrics somewhere
        ...         pass
    """

    def log(self, result: "MetricResult") -> None:
        """
        Log the metric result to some destination.

        Args:
            result: The MetricResult containing computed metrics.
        """
        ...
