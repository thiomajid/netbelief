import json
import typing as tp
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseScaler(ABC):
    """
    Base class for multi-dimensional scalers with save/load functionality.

    Provides common interface for fitting, transforming, and persisting scaler parameters.
    """

    def __init__(self):
        self.is_fitted_ = False

    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaseScaler":
        """Compute statistics from data."""
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply transformation to data."""
        pass

    @abstractmethod
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse transformation."""
        pass

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    @abstractmethod
    def _get_params_dict(self) -> dict:
        """Get scaler parameters as a dictionary for serialization."""
        pass

    @abstractmethod
    def _set_params_from_dict(self, params: dict) -> None:
        """Set scaler parameters from a dictionary."""
        pass

    def save(self, path: str | Path) -> None:
        """
        Save scaler parameters to JSON file.

        Args:
            path: File path where parameters will be saved

        Raises:
            RuntimeError: If scaler has not been fitted yet
        """
        if not self.is_fitted_:
            raise RuntimeError("Cannot save unfitted scaler. Call fit() first.")

        path = Path(path)
        if not path.parent.exists():
            raise FileNotFoundError(f"Scaler parent directory not found: {path}")

        params = self._get_params_dict()

        # Convert numpy arrays to lists for JSON serialization
        serializable_params = {}
        for key, value in params.items():
            if isinstance(value, np.ndarray):
                serializable_params[key] = {
                    "__type__": "ndarray",
                    "data": value.tolist(),
                    "dtype": str(value.dtype),
                    "shape": value.shape,
                }
            else:
                serializable_params[key] = value

        with open(path, "w") as f:
            json.dump(serializable_params, f, indent=2)

    def load(self, path: str | Path) -> "BaseScaler":
        """
        Load scaler parameters from JSON file.

        Args:
            path: File path to load parameters from

        Returns:
            self (for chaining)

        Raises:
            FileNotFoundError: If file does not exist
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Scaler file not found: {path}")

        with open(path, "r") as f:
            params = json.load(f)

        # Convert lists back to numpy arrays
        deserialized_params = {}
        for key, value in params.items():
            if isinstance(value, dict) and value.get("__type__") == "ndarray":
                array = np.array(value["data"], dtype=value["dtype"])
                deserialized_params[key] = array.reshape(value["shape"])
            else:
                deserialized_params[key] = value

        self._set_params_from_dict(deserialized_params)
        self.is_fitted_ = True

        return self


class MultiDimStandardScaler(BaseScaler):
    """
    Standard scaler for multi-dimensional arrays (3D, 4D, etc.).

    Computes mean and std along specified axes and applies standardization:
        x_scaled = (x - mean) / std

    Args:
        axis: Axis or axes along which to compute statistics.
              For time-series data with shape (batch, devices, metrics, time),
              use axis=(0, 3) to normalize across batch and time dimensions.
        epsilon: Small constant for numerical stability.

    Example:
        >>> # 3D array: (num_devices, num_metrics, timesteps)
        >>> scaler = MultiDimStandardScaler(axis=2)  # normalize over time
        >>> data_scaled = scaler.fit_transform(data)
        >>> data_original = scaler.inverse_transform(data_scaled)

        >>> # 4D array: (num_series, num_devices, num_metrics, lookback)
        >>> scaler = MultiDimStandardScaler(axis=(0, 3))  # normalize over series and time
        >>> data_scaled = scaler.fit_transform(data)
    """

    def __init__(
        self,
        axis: int | tuple[int, ...] = -1,
        epsilon: float = 1e-7,
    ):
        super().__init__()
        self.axis = axis if isinstance(axis, tuple) else (axis,)
        self.epsilon = epsilon

        self.mean_: tp.Optional[np.ndarray] = None
        self.std_: tp.Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "MultiDimStandardScaler":
        """
        Compute mean and standard deviation along specified axes.
        """
        # Compute statistics along specified axes
        self.mean_ = np.mean(X, axis=self.axis, keepdims=True)
        self.std_ = np.std(X, axis=self.axis, keepdims=True)

        # Prevent division by zero
        self.std_ = np.maximum(self.std_, self.epsilon)

        self.is_fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Scale data using computed statistics.
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "Scaler must be fitted before transforming data. Call fit() first."
            )

        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit scaler and transform data in one step.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Convert scaled data back to original scale.
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "Scaler must be fitted before inverse transforming. Call fit() first."
            )

        return X * self.std_ + self.mean_

    def _get_params_dict(self) -> dict:
        """Get scaler parameters for serialization."""
        return {
            "scaler_type": "MultiDimStandardScaler",
            "axis": self.axis,
            "epsilon": self.epsilon,
            "mean": self.mean_,
            "std": self.std_,
        }

    def _set_params_from_dict(self, params: dict) -> None:
        """Set scaler parameters from deserialized dictionary."""
        if params.get("scaler_type") != "MultiDimStandardScaler":
            raise ValueError(
                f"Invalid scaler type: {params.get('scaler_type')}. "
                "Expected MultiDimStandardScaler."
            )

        self.axis = tuple(params["axis"])
        self.epsilon = params["epsilon"]
        self.mean_ = params["mean"]
        self.std_ = params["std"]

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted_ else "not fitted"
        return (
            f"MultiDimStandardScaler(axis={self.axis}, epsilon={self.epsilon}, "
            f"status={status})"
        )


class MultiDimMinMaxScaler(BaseScaler):
    """uper().__init__()
        self.axis = axis if isinstance(axis, tuple) else (axis,)
        self.feature_range = feature_range
        self.epsilon = epsilon

        self.min_: tp.Optional[np.ndarray] = None
        self.max_: tp.Optional[np.ndarray] = None
        self.scale_: tp.Optional[float] = None
        self.data_range_: tp.Optional[np.ndarray] = Nonng which to compute min/max statistics.
        feature_range: Desired range of transformed data (min, max).
        epsilon: Small constant for numerical stability.

    Example:
        >>> # 3D array: (num_devices, num_metrics, timesteps)
        >>> scaler = MultiDimMinMaxScaler(axis=2, feature_range=(0, 1))
        >>> data_scaled = scaler.fit_transform(data)
    """

    def __init__(
        self,
        axis: int | tuple[int, ...] = -1,
        feature_range: tuple[float, float] = (0, 1),
        epsilon: float = 1e-7,
    ):
        self.axis = axis if isinstance(axis, tuple) else (axis,)
        self.feature_range = feature_range
        self.epsilon = epsilon

        self.min_: tp.Optional[np.ndarray] = None
        self.max_: tp.Optional[np.ndarray] = None
        self.scale_: tp.Optional[float] = None
        self.is_fitted_ = False

    def fit(self, X: np.ndarray) -> "MultiDimMinMaxScaler":
        """Compute min and max along specified axes."""
        self.min_ = np.min(X, axis=self.axis, keepdims=True)
        self.max_ = np.max(X, axis=self.axis, keepdims=True)

        # Prevent division by zero
        data_range = self.max_ - self.min_
        data_range = np.maximum(data_range, self.epsilon)

        # Compute scale for target range
        feature_min, feature_max = self.feature_range
        self.scale_ = feature_max - feature_min
        self.data_range_ = data_range

    def _get_params_dict(self) -> dict:
        """Get scaler parameters for serialization."""
        return {
            "scaler_type": "MultiDimMinMaxScaler",
            "axis": self.axis,
            "epsilon": self.epsilon,
            "feature_range": self.feature_range,
            "min": self.min_,
            "max": self.max_,
            "scale": self.scale_,
            "data_range": self.data_range_,
        }

    def _set_params_from_dict(self, params: dict) -> None:
        """Set scaler parameters from deserialized dictionary."""
        if params.get("scaler_type") != "MultiDimMinMaxScaler":
            raise ValueError(
                f"Invalid scaler type: {params.get('scaler_type')}. "
                "Expected MultiDimMinMaxScaler."
            )

        self.axis = tuple(params["axis"])
        self.epsilon = params["epsilon"]
        self.feature_range = tuple(params["feature_range"])
        self.min_ = params["min"]
        self.max_ = params["max"]
        self.scale_ = params["scale"]
        self.data_range_ = params["data_range"]
        self.is_fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale data to feature_range."""
        if not self.is_fitted_:
            raise RuntimeError("Scaler must be fitted before transforming data.")

        feature_min, _ = self.feature_range
        X_std = (X - self.min_) / self.data_range_
        return X_std * self.scale_ + feature_min

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit scaler and transform data in one step."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Convert scaled data back to original scale."""
        if not self.is_fitted_:
            raise RuntimeError("Scaler must be fitted before inverse transforming.")

        feature_min, _ = self.feature_range
        X_std = (X - feature_min) / self.scale_
        return X_std * self.data_range_ + self.min_

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted_ else "not fitted"
        return (
            f"MultiDimMinMaxScaler(axis={self.axis}, feature_range={self.feature_range}, "
            f"status={status})"
        )
