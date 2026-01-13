"""
Internal format adapter - accepts ODData directly for in-memory evaluation.
"""
from foresight_metrics.tasks.object_detection.types import ODData


class InternalFormat:
    """
    Format adapter that accepts ODData directly.

    Use this when you've already constructed ODData in memory
    (e.g., from running inference in a loop).

    Example:
        >>> adapter = InternalFormat()
        >>> data = adapter.load(od_data, od_data)  # Same object for both
    """

    name = "internal"

    def load(self, ground_truth: ODData, predictions: ODData) -> ODData:
        """
        Pass-through loader for ODData.

        Since ODData already contains both GT and predictions,
        we expect the same ODData object for both arguments.

        Args:
            ground_truth: ODData object (or the same object)
            predictions: ODData object (typically same as ground_truth)

        Returns:
            The ODData object unchanged.
        """
        # If same object passed for both, just return it
        if ground_truth is predictions:
            return ground_truth
        
        # If different objects, prefer ground_truth (which should be complete)
        return ground_truth
