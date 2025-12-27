import typing as tp

import grain.python as grain
import numpy as np
import polars as pl
from einops import rearrange

from src.data.scaler import MultiDimStandardScaler
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


def convert_dataframe_to_numpy(
    df: pl.DataFrame,
    metrics: tp.Sequence[str],
    logger: LoguruLogger,
):
    device_arrays: list[np.ndarray] = []
    for row in df.iter_rows(named=True):
        # stack metric columns into a 2D array: (num_metrics, timesteps)
        device_data = np.stack([np.array(row[col]) for col in metrics], axis=0)
        device_arrays.append(device_data)

    # find the length of real data for each device (before first NaN)
    real_data_lengths = []
    for arr in device_arrays:
        # check if any metric has NaN at each timestep
        has_nan = np.isnan(arr).any(axis=0)
        if has_nan.any():
            first_nan_idx = np.where(has_nan)[0][0]
            real_data_lengths.append(first_nan_idx)
        else:
            real_data_lengths.append(arr.shape[1])

    min_real_length = min(real_data_lengths)
    num_devices = len(device_arrays)
    num_metrics = len(metrics)

    logger.info(f"Total devices: {num_devices}")
    logger.warning(
        f"Real data lengths: min={min(real_data_lengths)}, max={max(real_data_lengths)}, mean={np.mean(real_data_lengths):.1f}"
    )
    logger.warning(f"Using min length: {min_real_length} (to ensure all real data)")

    device_arrays_truncated = [arr[:, :min_real_length] for arr in device_arrays]
    data_3d = np.stack(
        device_arrays_truncated, axis=0
    )  # (num_devices, num_metrics, timesteps)

    if np.isnan(data_3d).any():
        logger.warning(
            f"Warning: Still have {np.isnan(data_3d).sum()} NaN values after truncation"
        )
    else:
        logger.info("Success: No NaN values in final array")

    # compute stats per metric
    nan_ratio_per_metric = np.isnan(data_3d).sum(axis=(0, 2)) / (
        num_devices * min_real_length
    )
    variance_per_metric = np.nanvar(data_3d, axis=(0, 2))
    std_per_metric = np.nanstd(data_3d, axis=(0, 2))
    mean_per_metric = np.nanmean(data_3d, axis=(0, 2))

    # Print detailed statistics per metric
    print(f"{'Metric':<30} {'NaN %':<10} {'Mean':<15} {'Std':<15} {'Variance':<15}")
    print("=" * 85)
    for idx, metric in enumerate(metrics):
        print(
            f"{metric:<30} {nan_ratio_per_metric[idx]:>7.2%}   "
            f"{mean_per_metric[idx]:>13.2f}   "
            f"{std_per_metric[idx]:>13.2f}   "
            f"{variance_per_metric[idx]:>13.2f}"
        )

    data_3d = np.nan_to_num(data_3d)
    return data_3d

    # threshold = 0.01
    # valid_metrics = nan_ratio_per_metric <= threshold
    # discarded_metrics = [
    #     metrics[i] for i, valid in enumerate(valid_metrics) if not valid
    # ]

    # if discarded_metrics:
    #     logger.warning(f"Discarded metrics ({len(discarded_metrics)}):")
    #     for metric in discarded_metrics:
    #         idx = metrics.index(metric)
    #         logger.warning(f"- {metric} ({nan_ratio_per_metric[idx]:.2%} missing)")

    #     # Filter to keep only valid metrics
    #     data_3d_filtered = data_3d[:, valid_metrics, :]
    #     kept_metric_names = [m for m in metrics if m not in discarded_metrics]
    # else:
    #     logger.info(f"All {num_metrics} metrics kept")
    #     data_3d_filtered = data_3d
    #     kept_metric_names = metrics


def create_lstm_dataloaders(
    data: np.ndarray,
    lookback: int,
    horizon: int,
    train_fraction: float,
    seed: int,
    worker_count: int,
    worker_buffer_size: int,
    drop_remainder: bool = True,
    train_operations: tp.Optional[tp.Sequence[grain.Transformation]] = None,
    eval_operations: tp.Optional[tp.Sequence[grain.Transformation]] = None,
):
    context, target = partition_logs(data, lookback=lookback, horizon=horizon)
    split_index = int(context.shape[0] * train_fraction)
    train_context, train_target = context[:split_index], target[:split_index]
    test_context, test_target = context[split_index:], target[split_index:]

    feature_scaler = MultiDimStandardScaler(axis=(0, 3))
    target_scaler = MultiDimStandardScaler(axis=(0, 3))

    train_context_scaled = feature_scaler.fit_transform(train_context)
    train_target_scaled = target_scaler.fit_transform(train_target)

    test_context_scaled = feature_scaler.transform(test_context)
    test_target_scaled = target_scaler.transform(test_target)

    train_source = MetricsDataSource(train_context_scaled, train_target_scaled)
    val_source = MetricsDataSource(test_context_scaled, test_target_scaled)

    train_loader = grain.DataLoader(
        data_source=train_source,
        operations=train_operations,
        worker_count=worker_count,
        worker_buffer_size=worker_buffer_size,
        sampler=grain.IndexSampler(
            num_records=len(train_source),
            shuffle=True,
            num_epochs=1,
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
            num_records=len(val_source),
            shuffle=False,
            num_epochs=1,
            seed=seed,
            shard_options=grain.ShardByJaxProcess(drop_remainder=drop_remainder),
        ),
    )

    return train_loader, val_loader, feature_scaler, target_scaler
