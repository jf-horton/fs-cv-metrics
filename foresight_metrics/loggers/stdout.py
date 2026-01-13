"""
StdoutLogger - Prints metrics to terminal.
"""
from foresight_metrics.results import MetricResult


class StdoutLogger:
    """
    Logger that prints metrics to standard output.

    Example:
        >>> from foresight_metrics.loggers import StdoutLogger
        >>> logger = StdoutLogger()
        >>> logger.log(result)  # Prints formatted metrics to terminal
    """

    def log(self, result: MetricResult) -> None:
        """
        Print the metric result to stdout.

        Args:
            result: The MetricResult to display.
        """
        print(result.pretty())
