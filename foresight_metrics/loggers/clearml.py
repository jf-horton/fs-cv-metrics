"""
ClearMLLogger - Logs metrics to a ClearML experiment tracker.
"""
from typing import TYPE_CHECKING, Any

from foresight_metrics.results import MetricResult

if TYPE_CHECKING:
    # Avoid hard dependency on clearml
    pass


class ClearMLLogger:
    """
    Logger that sends metrics to a ClearML task.

    This logger integrates with ClearML experiment tracking to log
    metrics as scalars that can be viewed in the ClearML web UI.

    Example:
        >>> from clearml import Task
        >>> from foresight_metrics.loggers import ClearMLLogger
        >>>
        >>> task = Task.init(project_name="my_project", task_name="eval")
        >>> logger = ClearMLLogger(task=task)
        >>> logger.log(result)  # Metrics appear in ClearML dashboard
    """

    def __init__(self, task: Any) -> None:
        """
        Initialize the ClearML logger.

        Args:
            task: A ClearML Task object. The task should already be
                  initialized before passing it here.
        """
        self.task = task

    def log(self, result: MetricResult) -> None:
        """
        Log metrics to the ClearML task.

        Logs each metric as a scalar with the task name as the title
        and metric name as the series.

        Args:
            result: The MetricResult containing metrics to log.
        """
        logger = self.task.get_logger()

        # Log aggregate metrics
        for name, value in result.metrics.items():
            if isinstance(value, (int, float)):
                logger.report_scalar(
                    title=result.task_name,
                    series=name,
                    value=value,
                    iteration=0,
                )

        # Log per-class metrics
        for cls_name, cls_metrics in result.per_class.items():
            for metric_name, value in cls_metrics.items():
                if isinstance(value, (int, float)):
                    logger.report_scalar(
                        title=f"{result.task_name}/{cls_name}",
                        series=metric_name,
                        value=value,
                        iteration=0,
                    )
