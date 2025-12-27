import math
from dataclasses import dataclass

from loguru import Logger


@dataclass
class TrainingConfig:
    # data stuff
    data_hub_id: str
    data_split: str
    lookback: int
    horizon: int
    train_fraction: float
    worker_count: int
    worker_buffer_size: int
    drop_remainder: int
    batch_size: int
    mask_prob: float

    dtype: str
    param_dtype: str

    num_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int

    # optimization
    metrics: tuple[str, ...]
    seed: int
    learning_rate: float
    weight_decay: float
    adam_beta1: float
    adam_beta2: float
    warmup_ratio: float
    max_grad_norm: float

    # logging and checkpoints
    logging_steps: int
    output_dir: str
    logging_dir: str
    run_name: str
    best_metric_key: str
    best_n_to_keep: int

    hub_model_id: str
    hub_token: str
    hub_private_repo: bool
    upload_message: str

    # array sharding
    mesh_shape: tuple[int, ...] = (1, 8)
    axis_names: tuple[str, ...] = ("dp", "tp")

    def __post_init__(self):
        assert 0 <= self.mask_prob <= 1, (
            f"mask_prob should be in [0, 1] but got {self.mask_prob}"
        )

        assert self.metrics is not None and len(self.metrics) >= 1

        self.mesh_shape = tuple(self.mesh_shape)
        self.axis_names = tuple(self.axis_names)
        self.metrics = tuple(self.metrics)


@dataclass
class TrainingSteps:
    train_batches: int
    eval_batches: int
    max_steps: int
    max_optimizer_steps: int
    steps_per_epoch: int
    optimizer_steps_per_epoch: int


def compute_training_steps(
    args: TrainingConfig,
    train_samples: int,
    eval_samples: int,
    logger: Logger,
) -> TrainingSteps:
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")

    # Calculate number of batches per epoch
    train_batches = train_samples // args.per_device_train_batch_size
    eval_batches = eval_samples // args.per_device_eval_batch_size

    logger.info(f"Batches per epoch - Train: {train_batches}, Eval: {eval_batches}")

    # Each batch is processed as one step
    steps_per_epoch = train_batches

    # Optimizer updates happen every gradient_accumulation_steps batches
    optimizer_steps_per_epoch = math.ceil(
        train_batches // args.gradient_accumulation_steps
    )

    logger.info(f"Steps per epoch: {steps_per_epoch}")
    logger.info(f"Optimizer steps per epoch: {optimizer_steps_per_epoch}")

    if optimizer_steps_per_epoch == 0:
        logger.warning(
            f"Number of batches per epoch ({train_batches}) is less than gradient_accumulation_steps ({args.gradient_accumulation_steps}). "
            "Effective optimizer steps per epoch is 0. Consider reducing accumulation steps or increasing dataset size."
        )

    max_steps = int(args.num_epochs * steps_per_epoch)
    max_optimizer_steps = int(args.num_epochs * optimizer_steps_per_epoch)

    return TrainingSteps(
        train_batches=train_batches,
        eval_batches=eval_batches,
        max_steps=max_steps,
        max_optimizer_steps=max_optimizer_steps,
        steps_per_epoch=steps_per_epoch,
        optimizer_steps_per_epoch=optimizer_steps_per_epoch,
    )
