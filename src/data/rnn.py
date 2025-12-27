import typing as tp

import grain.python as grain
import numpy as np
from einops import rearrange

from src.utils.types import LoguruLogger


def partition_logs(
    data: np.ndarray,
    lookback: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slices the input data to prepare batches for an RNN to train on

    :param data: Input data to partition of shape [num_devices, metrics, timestep]
    :type data: np.ndarray
    :param lookback: The context provided to the RNN to predict H future steps
    :type lookback: int
    :param horizon: The forecasting horizon that the RNN will predict
    :type horizon: int

    Returns
    -------
    Overlapping input time series of shape (num_series, num_devices, num_metrics, lookback)
    and their targets of shape (num_series, num_devices, num_metrics, horizon).
    """

    num_devices, num_metrics, T = data.shape
    time_major = rearrange(data, "d m t -> t (d m)")

    # last sample must satisfy start + lookback + horizon <= T
    # therefore, start <= T - lookback - horizon
    # the max value for start is T - lookback - horizon, therefore add 1
    indices = (
        np.arange(lookback + horizon)[None, :]
        + np.arange(T - lookback - horizon + 1)[:, None]
    )

    windows = time_major[indices]  # (N, K+H, F) with N being the number of windows
    series = windows[:, :lookback, :]  # (N, K, F) -> timesteps [0:K]
    targets = windows[:, lookback:, :]  # (N, H, F) -> timesteps [K:K+H]

    series = rearrange(series, "n t (d m) -> n d m t", d=num_devices)
    targets = rearrange(targets, "n t (d m) -> n d m t", d=num_devices)

    return series, targets


class MetricsDataSource(grain.RandomAccessDataSource):
    def __init__(self, context: np.ndarray, target: np.ndarray):
        self.context = context
        self.target = target

    def __len__(self):
        return self.context.shape[0]

    def __getitem__(self, record_key):
        return self.context[record_key], self.target[record_key]


def create_lstm_dataloaders(
    data: np.ndarray,
    lookback: int,
    horizon: int,
    train_fraction: float,
    seed: int,
    worker_count: int,
    worker_buffer_size: int,
    logger: LoguruLogger,
    drop_remainder: bool = True,
    train_operations: tp.Optional[tp.Sequence[grain.Transformation]] = None,
    eval_operations: tp.Optional[tp.Sequence[grain.Transformation]] = None,
):
    context, target = partition_logs(data, lookback=lookback, horizon=horizon)
    split_index = int(context.shape[0] * train_fraction)
    train_context, train_target = context[:split_index], target[:split_index]
    val_context, val_target = context[split_index:], target[split_index:]

    train_source = MetricsDataSource(train_context, train_target)
    val_source = MetricsDataSource(val_context, val_target)

    train_loader = grain.DataLoader(
        data_source=train_source,
        operations=train_operations,
        worker_count=worker_count,
        worker_buffer_size=worker_buffer_size,
        sampler=grain.IndexSampler(
            num_records=len(train_source),
            shuffle=True,
            seed=seed,
            shard_options=grain.ShardByJaxProcess(drop_remainder=drop_remainder),
        ),
    )

    val_loader = grain.DataLoader(
        data_source=val_source,
        operations=eval_operations,
        worker_count=worker_count,
        worker_buffer_size=worker_buffer_size,
        sampler=grain.IndexSampler(
            num_records=len(train_source),
            shuffle=False,
            seed=seed,
            shard_options=grain.ShardByJaxProcess(drop_remainder=drop_remainder),
        ),
    )

    return train_loader, val_loader
