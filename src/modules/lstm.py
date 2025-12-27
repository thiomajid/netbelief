import typing as tp
from dataclasses import dataclass
from functools import partial

import chex
import jax
import jax.numpy as jnp
from einops import rearrange
from flax import nnx
from flax.nnx import initializers

from src.modules.shared import GroupedQueryAttention
from src.utils.types import EncodedBelief, ForecasterOutput, ShardingRule


@dataclass(unsafe_hash=True, eq=True)
class LSTMForecasterConfig:
    num_metrics: int
    hidden_features: int
    num_heads: int = 4
    num_kv_heads: int = 2
    horizon: int = 1
    head_bias: bool = False
    normalize_qk: bool = True
    use_device_mixer: bool = True
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9, 0.95)

    def __post_init__(self):
        self.quantiles = tuple(self.quantiles)


@dataclass(unsafe_hash=True, eq=True)
class LSTMForecasterShardings:
    norm: ShardingRule = (None,)
    lstm_input_kernel: ShardingRule = (None, "tp")
    lstm_recurrent_kernel: ShardingRule = (None, "tp")
    lstm_bias: ShardingRule = (None,)
    attn_qkv: ShardingRule = (None, "tp")
    attn_output: ShardingRule = ("tp", None)
    attn_norm: ShardingRule = ("tp",)
    head_kernel: ShardingRule = ("tp", None)
    head_bias: ShardingRule = (None,)


class LSTMForecaster(nnx.Module):
    def __init__(
        self,
        config: LSTMForecasterConfig,
        *,
        shardings: LSTMForecasterShardings,
        rngs: nnx.Rngs,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
    ):
        self.num_metrics = config.num_metrics
        self.horizon = config.horizon
        self.use_device_mixer = config.use_device_mixer

        self.quantiles = config.quantiles
        self.num_quantiles = len(config.quantiles)

        self.norm = nnx.LayerNorm(
            num_features=config.num_metrics,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            scale_init=nnx.with_partitioning(
                initializers.ones_init(),
                sharding=shardings.norm,
            ),
            bias_init=nnx.with_partitioning(
                initializers.zeros_init(),
                sharding=shardings.norm,
            ),
        )

        cell = nnx.OptimizedLSTMCell(
            in_features=config.num_metrics,
            hidden_features=config.hidden_features,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=nnx.with_partitioning(
                initializers.lecun_normal(),
                sharding=shardings.lstm_input_kernel,
            ),
            recurrent_kernel_init=nnx.with_partitioning(
                initializers.orthogonal(),
                sharding=shardings.lstm_recurrent_kernel,
            ),
            bias_init=nnx.with_partitioning(
                initializers.zeros_init(),
                sharding=shardings.lstm_bias,
            ),
        )

        self.rnn = nnx.RNN(cell=cell, rngs=rngs)

        self.mixer: tp.Optional[GroupedQueryAttention]

        if config.use_device_mixer:
            self.mixer = GroupedQueryAttention(
                in_features=config.hidden_features,
                out_features=config.hidden_features,
                num_heads=config.num_heads,
                num_kv_heads=config.num_kv_heads,
                rngs=rngs,
                dtype=dtype,
                param_dtype=param_dtype,
                normalize_qk=config.normalize_qk,
                implementation="xla",
                is_causal=False,
                qkv_kernel_init=nnx.with_partitioning(
                    initializers.lecun_normal(),
                    sharding=shardings.attn_qkv,
                ),
                out_kernel_init=nnx.with_partitioning(
                    initializers.lecun_normal(),
                    sharding=shardings.attn_output,
                ),
                scale_init=nnx.with_partitioning(
                    initializers.ones_init(),
                    sharding=shardings.attn_norm,
                ),
            )
        else:
            self.mixer = nnx.data(None)

        Linear = partial(
            nnx.Linear,
            in_features=config.hidden_features,
            out_features=config.num_metrics * config.horizon,
            use_bias=config.head_bias,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=nnx.with_partitioning(
                initializers.lecun_normal(),
                sharding=shardings.head_kernel,
            ),
            bias_init=nnx.with_partitioning(
                initializers.zeros_init(),
                sharding=shardings.head_bias,
            ),
        )

        self.point_head = Linear()
        self.quantile_head = Linear(
            out_features=config.num_metrics * config.horizon * self.num_quantiles
        )

    def encode(
        self,
        series: jax.Array,
        initial_carry: tp.Optional[jax.Array] = None,
    ):
        chex.assert_rank(series, 4)
        if initial_carry is not None:
            chex.assert_rank(initial_carry, 4)

        batch, devices, metrics, time = series.shape
        flat_series = rearrange(series, "b d m t -> (b d) t m")
        flat_carry = (
            rearrange(initial_carry, "b d m t -> (b d) t m")
            if initial_carry is not None
            else None
        )

        hidden = self.rnn(
            self.norm(flat_series),
            initial_carry=flat_carry,
        )  # (batch*devices, time, hidden)

        # reshape back to device axis
        # (batch * devices, 1, hidden) -> (batch, devices, hidden)
        last_hidden_state = hidden[:, -1, :].reshape(batch, devices, -1)

        if not self.use_device_mixer:
            return EncodedBelief(
                belief=last_hidden_state,
                last_hidden_state=last_hidden_state,
            )

        belief, attention_ouput = self.mixer(last_hidden_state)
        return EncodedBelief(
            belief=belief,
            last_hidden_state=last_hidden_state,
            attention_ouput=attention_ouput,
        )

    def __call__(
        self,
        series: jax.Array,
        initial_carry: tp.Optional[jax.Array] = None,
    ):
        encoding = self.encode(series, initial_carry)
        belief = encoding.belief

        prediction = self.point_head(belief)
        prediction = rearrange(prediction, "b d (m h) -> b d m h", h=self.horizon)

        quantiles = self.quantile_head(belief)
        quantiles = rearrange(
            quantiles,
            "b d (m h q) -> q b d m h",
            h=self.horizon,
            q=self.num_quantiles,
        )

        return ForecasterOutput(
            encoding=encoding,
            point_predictions=prediction,
            quantile_predictions=quantiles,
        )
