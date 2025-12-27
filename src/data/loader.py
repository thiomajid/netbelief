"""Data loading utilities for LSTM training."""

import typing as tp
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


def load_csv_data(
    data_path: str | Path,
    exclude_columns: tp.Optional[list[str]] = None,
) -> np.ndarray:
    """
    Load data from CSV file and convert to numpy array.

    Args:
        data_path: Path to CSV file
        exclude_columns: List of column names to exclude (e.g., timestamp, flow_id)

    Returns:
        numpy array of shape (num_devices, num_metrics, timesteps)
    """
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)

    # Default columns to exclude
    if exclude_columns is None:
        exclude_columns = ["timestamp", "flow_id", "interval_start", "interval_end"]

    # Remove excluded columns
    metric_columns = [col for col in df.columns if col not in exclude_columns]
    logger.info(f"Using {len(metric_columns)} metric columns: {metric_columns}")

    # Extract metric data
    data = df[metric_columns].values  # Shape: (timesteps, num_metrics)

    # Handle NaN values
    if np.isnan(data).any():
        logger.warning("Found NaN values in data, replacing with 0")
        data = np.nan_to_num(data, nan=0.0)

    # Normalize or standardize if needed
    # For now, just return the raw data
    # You may want to add normalization here

    logger.info(f"Loaded data with shape: {data.shape}")

    return data


def prepare_network_data(
    data_path: str | Path,
    num_devices: int = 16,  # Default for network topology
) -> np.ndarray:
    """
    Prepare network data for LSTM training.

    Args:
        data_path: Path to CSV file
        num_devices: Number of network devices

    Returns:
        numpy array of shape (num_devices, num_metrics, timesteps)
    """
    data = load_csv_data(data_path)
    timesteps, num_metrics = data.shape

    # Reshape to (num_devices, num_metrics, timesteps)
    # Assuming data is organized sequentially by device
    # You may need to adjust this based on your actual data structure

    # For now, we'll treat each timestep as independent and replicate across devices
    # This is a simplified approach - adjust based on your actual data organization
    data_per_device = data.T  # (num_metrics, timesteps)

    # Replicate data for multiple devices if needed
    # This is a placeholder - you should adjust based on actual data structure
    if num_devices > 1:
        logger.warning(
            f"Replicating data across {num_devices} devices. "
            "Adjust this based on your actual data organization."
        )
        data_reshaped = np.stack([data_per_device] * num_devices, axis=0)
    else:
        data_reshaped = data_per_device[None, :, :]  # Add device dimension

    logger.info(f"Prepared data with shape: {data_reshaped.shape}")

    return data_reshaped
