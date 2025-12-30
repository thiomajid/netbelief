"""Base metrics reporter interface for training."""

import typing as tp
from abc import ABC, abstractmethod
from pathlib import Path

import jax
import numpy as np

from src.utils.types import LoguruLogger


class MetricsReporter(ABC):
    """Abstract base class for metrics reporters."""

    def __init__(
        self, log_dir: tp.Union[str, Path], logger: LoguruLogger, name: str = "train"
    ):
        """Initialize metrics reporter.

        Args:
            log_dir: Directory to save logs
            logger: Loguru logger instance
            name: Name for this reporter instance
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.logger = logger

    def maybe_scalar(self, value: tp.Union[jax.Array, np.ndarray]):
        if isinstance(value, (jax.Array, np.ndarray)):
            value = float(value.item())
        elif not isinstance(value, (int, float)):
            value = float(value)

        return value

    @abstractmethod
    def log_scalar(
        self,
        tag: str,
        value: tp.Union[float, jax.Array, np.ndarray],
        step: int,
    ):
        """Log a scalar value.

        Args:
            tag: Name of the scalar
            value: Scalar value to log
            step: Global step
        """
        pass

    @abstractmethod
    def log_scalars(
        self,
        tag_scalar_dict: tp.Dict[str, tp.Union[float, jax.Array, np.ndarray]],
        step: int,
    ):
        """Log multiple scalars at once.

        Args:
            tag_scalar_dict: Dictionary of tag -> scalar value
            step: Global step
        """
        pass

    @abstractmethod
    def log_histogram(
        self,
        tag: str,
        values: tp.Union[jax.Array, np.ndarray],
        step: int,
    ):
        """Log a histogram of values.

        Args:
            tag: Name for the histogram
            values: Values to create histogram from
            step: Global step
        """
        pass

    @abstractmethod
    def log_figure(
        self,
        tag: str,
        figure: np.ndarray,
        step: int,
    ):
        """Log a matplotlib figure as an image.

        Args:
            tag: Name for the figure
            figure: Matplotlib figure object
            step: Global step
        """
        pass

    @abstractmethod
    def log_learning_rate(self, lr: float, step: int):
        """Log learning rate.

        Args:
            lr: Learning rate value
            step: Global step
        """
        pass

    @abstractmethod
    def log_gradients(self, grads: tp.Dict[str, tp.Any], step: int):
        """Log gradient statistics.

        Args:
            grads: Dictionary of gradients
            step: Global step
        """
        pass

    @abstractmethod
    def close(self):
        """Close the reporter and cleanup resources."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
