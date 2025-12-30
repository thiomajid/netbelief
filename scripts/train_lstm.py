import sys

from src.training.reporter.base_reporter import MetricsReporter
from src.training.reporter.trackio_reporter import TrackioReporter
from src.utils.types import ForecasterInput

sys.path.append("../")


import os
import typing as tp
from pathlib import Path
from pprint import pprint

import grain.python as grain
import hydra
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from datasets import load_dataset
from flax import nnx
from huggingface_hub import snapshot_download
from jax.sharding import AxisType, NamedSharding
from jax.sharding import PartitionSpec as P
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from orbax.checkpoint import checkpoint_managers
from transformers import HfArgumentParser

from src.data.rnn import convert_dataframe_to_numpy, create_lstm_dataloaders
from src.data.transformations import DeviceMaskingTransform
from src.modules.lstm import (
    LSTMForecaster,
    LSTMForecasterConfig,
    LSTMForecasterShardings,
)
from src.training.arguments import TrainingConfig, compute_training_steps
from src.training.callback import (
    CheckpointCallback,
    PlotForecastCallback,
    PushToHubCallback,
)
from src.training.loss import dual_head_loss
from src.training.module import (
    count_parameters,
    load_sharded_checkpoint_state,
    str2dtype,
)
from src.training.reporter.tensorboard_reporter import TensorBoardReporter
from src.training.trainer import Trainer
from src.utils.array import log_node_devices_stats


def loss_fn(
    model: LSTMForecaster,
    batch: tuple[jax.Array, jax.Array],
    quantiles: jax.Array,
    point_weight: jax.Array,
):
    series, targets = batch
    output = model(series)
    loss_struct = dual_head_loss(
        point_predictions=output.point_predictions,
        quantile_predictions=output.quantile_predictions,
        targets=targets,
        quantiles=quantiles,
        point_weight=point_weight,
    )

    return loss_struct.total, loss_struct


@nnx.jit
def train_step(
    model: LSTMForecaster,
    batch: tuple[jax.Array, jax.Array],
    quantiles: jax.Array,
    point_weight: jax.Array,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
):
    # with jax.debug_nans(True):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, loss_struct), grads = grad_fn(model, batch, quantiles, point_weight)

    # Debugging NaNs
    # jax.debug.print("loss: {loss}", loss=loss)
    # is_nan_loss = jnp.isnan(loss)
    # jax.debug.print("is_nan_loss: {is_nan_loss}", is_nan_loss=is_nan_loss)

    grad_norm = optax.global_norm(grads)
    # jax.debug.print("grad_norm: {grad_norm}", grad_norm=grad_norm)
    # is_nan_grad = jnp.isnan(grad_norm)
    # jax.debug.print("is_nan_grad: {is_nan_grad}", is_nan_grad=is_nan_grad)

    optimizer.update(model, grads)

    metrics.update(
        loss=loss,
        grad_norm=grad_norm,
        mse=loss_struct.mse,
        mae=loss_struct.mae,
        rmse=loss_struct.rmse,
        q_loss=loss_struct.q_loss,
    )

    return loss, grads, grad_norm


@nnx.jit
def eval_step(
    model: LSTMForecaster,
    batch: tuple[jax.Array, jax.Array],
    quantiles: jax.Array,
    point_weight: jax.Array,
    metrics: nnx.MultiMetric,
):
    loss, loss_struct = loss_fn(model, batch, quantiles, point_weight)

    metrics.update(
        loss=loss,
        mse=loss_struct.mse,
        mae=loss_struct.mae,
        rmse=loss_struct.rmse,
        q_loss=loss_struct.q_loss,
    )

    return loss


@hydra.main(
    config_path="../configs",
    config_name="train_lstm",
    version_base="1.2",
)
def main(cfg: DictConfig):
    logger.info("Starting LSTM training...")

    parser = HfArgumentParser(TrainingConfig)

    # Load trainer arguments from YAML file
    args = parser.parse_dict(OmegaConf.to_container(cfg["trainer"], resolve=True))[0]  # ty: ignore
    args = tp.cast(TrainingConfig, args)

    os.environ["HF_TOKEN"] = args.hub_token

    OUTPUT_DIR = Path(args.output_dir)
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True)

    # Create model config from cfg
    log_node_devices_stats(logger)
    dtype = str2dtype(args.dtype)
    param_dtype = str2dtype(args.param_dtype)
    logger.info(
        f"Creating LSTM model with dtype={dtype} and param_dtype={param_dtype}..."
    )

    mesh_shape = tuple(args.mesh_shape)
    axis_names = tuple(args.axis_names)
    mesh = jax.make_mesh(
        mesh_shape,
        axis_names,
        axis_types=(AxisType.Auto, AxisType.Auto),
    )
    jax.set_mesh(mesh)
    logger.warning(f"Set global mesh with {mesh}")

    rngs = nnx.Rngs(args.seed)

    model_config_dict = OmegaConf.to_container(cfg["model"], resolve=True)
    pprint(model_config_dict)

    # Parse model config typesafely
    model_config_parser = HfArgumentParser(LSTMForecasterConfig)
    model_config = model_config_parser.parse_dict(model_config_dict)[0]
    model_config = tp.cast(LSTMForecasterConfig, model_config)
    logger.info(f"Parsed model config: {model_config}")

    assert model_config.num_metrics == len(args.metrics), (
        "The model must be trained to forecast the same number of selected target columns from the dataset"
    )

    sharding_parser = HfArgumentParser(LSTMForecasterShardings)
    shardings_dict = cfg.get("shardings", None)
    shardings = None

    if shardings_dict is not None:
        shardings = sharding_parser.parse_dict(
            OmegaConf.to_container(shardings_dict, resolve=True)
        )[0]
        shardings = tp.cast(LSTMForecasterShardings, shardings)
    else:
        shardings = LSTMForecasterShardings()

    with jax.set_mesh(mesh):
        model = LSTMForecaster(
            config=model_config,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            shardings=shardings,
        )

    logger.info(f"Model parameters: {count_parameters(model)}")
    log_node_devices_stats(logger)

    if cfg.get("resume_from_checkpoint", False):
        logger.info(f"Resuming from checkpoint from {cfg['checkpoint_hub_url']}")
        save_dir = Path(cfg["checkpoint_save_dir"])

        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id=cfg["checkpoint_hub_url"],
            local_dir=save_dir,
            token=args.hub_token,
            revision=cfg.get("checkpoint_revision", "main"),
        )

        ckpt_path = save_dir / "model_checkpoint/default"
        ckpt_path = ckpt_path.absolute()

        load_sharded_checkpoint_state(
            model=model,
            checkpoint_path=ckpt_path,
            mesh=mesh,
            logger=logger,
        )

        logger.info("loaded checkpoint state")

    logger.info("Setting model in training mode")
    model.train()

    # Load data
    logger.info(f"Loading data from {args.data_hub_id}")
    hub_data = load_dataset(
        args.data_hub_id, split=args.data_split, token=args.hub_token
    )
    logger.info(f"Loaded data with shape: {hub_data.shape}")

    data = hub_data.select_columns(args.metrics).to_polars()
    logger.info(f"Dataset columns are {data.columns}")

    data = convert_dataframe_to_numpy(df=data, metrics=args.metrics, logger=logger)
    logger.info(f"Dataset in numpy format has shape {data.shape}")

    # Create data transforms pipeline
    train_transforms = [
        grain.Batch(batch_size=args.per_device_train_batch_size, drop_remainder=True),
        DeviceMaskingTransform(mask_prob=args.mask_prob),
    ]

    eval_transforms = [
        grain.Batch(batch_size=args.per_device_eval_batch_size, drop_remainder=True),
        DeviceMaskingTransform(mask_prob=0.0),  # no masking
    ]

    train_loader, eval_loader, feature_scaler, target_scaler = create_lstm_dataloaders(
        data=data,
        lookback=args.lookback,
        horizon=args.horizon,
        train_fraction=args.train_fraction,
        seed=args.seed,
        worker_count=args.worker_count,
        worker_buffer_size=args.worker_buffer_size,
        drop_remainder=args.drop_remainder,
        train_operations=train_transforms,
        eval_operations=eval_transforms,
    )

    # save scalers for later usage
    feature_scaler.save(OUTPUT_DIR / "feature_scaler.json")
    target_scaler.save(OUTPUT_DIR / "target_scaler.json")
    logger.info(f"Saved scalers params in the output directory at {OUTPUT_DIR}")

    # Setup the training loop
    num_train_samples = len(train_loader._data_source)
    num_eval_samples = len(eval_loader._data_source)

    logger.info(f"Dataset sizes - Train: {num_train_samples}, Eval: {num_eval_samples}")
    train_steps_config = compute_training_steps(
        args,
        num_train_samples,
        num_eval_samples,
        logger,
    )

    max_steps = train_steps_config.max_steps
    max_optimizer_steps = train_steps_config.max_optimizer_steps

    # Set default warmup_ratio if not provided
    if not hasattr(args, "warmup_ratio"):
        args.warmup_ratio = 0.05
        logger.warning(
            f"warmup_ratio not found in config, defaulting to {args.warmup_ratio}"
        )

    # Use optimizer steps for learning rate schedule (not micro-batch steps)
    warmup_steps = int(args.warmup_ratio * max_optimizer_steps)
    logger.info(
        f"Calculated warmup steps: {warmup_steps} ({args.warmup_ratio=}, max_optimizer_steps={max_optimizer_steps})"
    )

    # linear warmup from 0 to peak learning rate
    warmup_schedule = optax.linear_schedule(
        init_value=0.0,
        end_value=args.learning_rate,
        transition_steps=warmup_steps,
    )

    # cosine decay from peak to end learning rate
    decay_steps = max_optimizer_steps - warmup_steps
    cosine_decay_schedule = optax.cosine_decay_schedule(
        init_value=args.learning_rate,
        decay_steps=decay_steps,
        alpha=0.2,  # end_value = init_value * alpha
    )

    # join the schedules: linear warmup -> cosine decay
    cosine_schedule = optax.join_schedules(
        schedules=[warmup_schedule, cosine_decay_schedule],
        boundaries=[warmup_steps],
    )

    logger.info(
        f"Using linear warmup + cosine decay schedule: 0.0 -> {args.learning_rate} (linear, {warmup_steps} steps) -> {args.learning_rate * 0.2} (cosine, {decay_steps} steps)"
    )

    # Optimizer
    optimizer_def = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optax.adamw(
            learning_rate=cosine_schedule,
            b1=args.adam_beta1,
            b2=args.adam_beta2,
            weight_decay=args.weight_decay,
        ),
    )

    optimizer_def = optax.MultiSteps(
        optimizer_def,
        every_k_schedule=args.gradient_accumulation_steps,
    )

    with jax.set_mesh(mesh):
        optimizer = nnx.Optimizer(model, optimizer_def, wrt=nnx.Param)  # ty: ignore

    # Metrics
    train_metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
        grad_norm=nnx.metrics.Average("grad_norm"),
        mse=nnx.metrics.Average("mse"),
        mae=nnx.metrics.Average("mae"),
        rmse=nnx.metrics.Average("rmse"),
        q_loss=nnx.metrics.Average("q_loss"),
    )

    eval_metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
        mse=nnx.metrics.Average("mse"),
        mae=nnx.metrics.Average("mae"),
        rmse=nnx.metrics.Average("rmse"),
        q_loss=nnx.metrics.Average("q_loss"),
    )

    # TensorBoard logger
    reporter: MetricsReporter
    if args.report_to == "tensorboard":
        reporter = TensorBoardReporter(
            log_dir=args.logging_dir,
            logger=logger,
            name=args.run_name,
        )
    else:
        reporter = TrackioReporter(
            log_dir=args.logging_dir,
            logger=logger,
            args=args,
        )

    # Checkpoint manager
    checkpoint_dir = Path(args.logging_dir).absolute()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_options = ocp.CheckpointManagerOptions(
        best_mode="min",
        create=True,
        preservation_policy=checkpoint_managers.BestN(
            get_metric_fn=lambda metrics: metrics[args.best_metric_key],
            n=args.best_n_to_keep,
            keep_checkpoints_without_metrics=False,
        ),
    )

    logger.info("Starting training loop...")
    logger.info(f"Num Epochs = {args.num_epochs}")
    logger.info(f"Micro Batch size = {args.per_device_train_batch_size}")
    logger.info(f"Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(
        f"Effective Batch size = {args.per_device_train_batch_size * args.gradient_accumulation_steps}"
    )
    logger.info(
        f"Total batches per epoch: Train - {train_steps_config.train_batches} && Eval - {train_steps_config.eval_batches}"
    )
    logger.info(f"Total steps = {max_steps}")
    logger.info(f"Total optimizer steps = {max_optimizer_steps}")

    DATA_SHARDING = NamedSharding(mesh, spec=P("dp", None))
    GENERATION_SAMPLE: ForecasterInput = next(iter(train_loader))

    SAMPLE_SERIES = jnp.array(GENERATION_SAMPLE.series)
    if SAMPLE_SERIES.shape[0] == 1:
        SAMPLE_SERIES = SAMPLE_SERIES.squeeze(0)
    SAMPLE_SERIES = jax.device_put(SAMPLE_SERIES, DATA_SHARDING)

    SAMPLE_TARGETS = jnp.array(GENERATION_SAMPLE.targets)
    if SAMPLE_TARGETS.shape[0] == 1:
        SAMPLE_TARGETS = SAMPLE_TARGETS.squeeze(0)
    SAMPLE_TARGETS = jax.device_put(SAMPLE_TARGETS, DATA_SHARDING)

    CALLBACKS = [
        CheckpointCallback(
            model=model,
            options=checkpoint_options,
            checkpoint_dir=checkpoint_dir,
            logger=logger,
        ),
        PushToHubCallback(logger=logger, args=args),
        PlotForecastCallback(
            model=model,
            reporter=reporter,
            series=SAMPLE_SERIES,
            targets=SAMPLE_TARGETS,
        ),
    ]

    QUANTILES = jnp.array(model_config.quantiles)

    trainer = Trainer(
        model=model,
        model_config=model_config,
        args=args,
        quantiles=QUANTILES,
        optimizer=optimizer,
        lr_scheduler=cosine_schedule,
        mesh=mesh,
        train_step_fn=train_step,
        eval_step_fn=eval_step,
        data_sharding=DATA_SHARDING,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        train_metrics=train_metrics,
        eval_metrics=eval_metrics,
        reporter=reporter,
        logger=logger,
        steps_config=train_steps_config,
        callbacks=CALLBACKS,
    )

    trainer.train()

    logger.info("Everything is done")


if __name__ == "__main__":
    main()
