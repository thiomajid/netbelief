import typing as tp

import jax
import numpy as np
from flax import struct
from loguru import logger as _logger

LoguruLogger = type[_logger]
ShardingRule = tuple[tp.Optional[str], ...]

ForecastingMode = tp.Literal["autoregressive", "multi-step"]
AttentionImpl = tp.Literal["xla", "cudnn"]


@struct.dataclass
class EncodedBelief:
    belief: jax.Array
    rnn_hidden_seq: jax.Array
    rnn_last_hidden_state: jax.Array
    attention_ouput: tp.Optional[jax.Array] = struct.field(default=None)


@struct.dataclass
class ForecasterInput:
    series: np.ndarray
    targets: np.ndarray
    mask: tp.Optional[np.ndarray] = struct.field(default=None)


@struct.dataclass
class ForecasterOutput:
    encoding: EncodedBelief
    point_predictions: jax.Array
    quantile_predictions: jax.Array


@struct.dataclass
class PlannerOutput:
    action_logits: jax.Array
    scores: jax.Array


@struct.dataclass
class ForecastingLossWithQuantiles:
    mse: jax.Array
    mae: jax.Array
    rmse: jax.Array
    q_loss: jax.Array
    quantile_losses: jax.Array
    total: jax.Array
