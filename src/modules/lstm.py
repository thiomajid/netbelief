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
    use_device_mixer: bool = False
    num_blocks: int = 2
    bidirectional: bool = False
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


class ForecasterBlock(nnx.Module):
    """
    Single block of LSTM + optional attention mixer.
    Can process sequences forward or backward while preserving temporal order.
    """

    def __init__(
        self,
        config: LSTMForecasterConfig,
        in_features: int,
        intermediate_features: int,
        *,
        shardings: LSTMForecasterShardings,
        rngs: nnx.Rngs,
        reverse: bool = False,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
    ):
        self.use_device_mixer = config.use_device_mixer
        self.reverse = reverse

        cell = nnx.OptimizedLSTMCell(
            in_features=in_features,
            hidden_features=intermediate_features,
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

        self.rnn = nnx.RNN(cell=cell, rngs=rngs, reverse=reverse)

        self.mixer_norm: tp.Optional[nnx.LayerNorm]
        self.mixer: tp.Optional[GroupedQueryAttention]

        if config.use_device_mixer:
            self.mixer_norm = nnx.LayerNorm(
                num_features=intermediate_features,
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

            self.mixer = GroupedQueryAttention(
                in_features=intermediate_features,
                out_features=intermediate_features,
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
            self.mixer_norm = nnx.data(None)
            self.mixer = nnx.data(None)

        self.down_proj = self.up_proj = nnx.Linear(
            in_features=intermediate_features,
            out_features=in_features,
            use_bias=False,
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

    def __call__(self, series: jax.Array) -> tuple[jax.Array, EncodedBelief]:
        """
        Process series through LSTM + attention.

        Args:
            series: (batch, devices, time, features)

        Returns:
            Tuple of:
            - Transformed sequence (batch, devices, time, out_features=in_features)
            - EncodedBelief with last hidden state info
        """
        chex.assert_rank(series, 4)

        batch, devices, time, features = series.shape
        flat_series = rearrange(series, "b d h t -> (b d) t h")
        hidden_seq = self.rnn(
            flat_series,
            reverse=self.reverse,
        )  # (b*d, t, intermediate)

        # (b*d, t, h) -> (b, d, t, h)
        hidden_seq = hidden_seq.reshape(batch, devices, time, -1)
        last_hidden_state = hidden_seq[:, :, -1, :]  # (b, d, intermediate)

        if not self.use_device_mixer:
            out_seq = self.down_proj(hidden_seq)
            return out_seq, EncodedBelief(
                belief=last_hidden_state,
                rnn_hidden_seq=hidden_seq,
                rnn_last_hidden_state=last_hidden_state,
            )

        normed_hidden = rearrange(self.mixer_norm(hidden_seq), "b d t h -> (b t) d h")
        mixer_output, attention_output = self.mixer(normed_hidden)
        mixer_output = rearrange(mixer_output, "(b t) d h -> b d t h", b=batch)
        belief = self.down_proj(hidden_seq + mixer_output)

        return belief, EncodedBelief(
            belief=belief,
            rnn_hidden_seq=hidden_seq,
            rnn_last_hidden_state=last_hidden_state,
            attention_ouput=attention_output,
        )


class BidirectionalForecasterBlock(nnx.Module):
    def __init__(
        self,
        config: LSTMForecasterConfig,
        in_features: int,
        intermediate_features: int,
        *,
        shardings: LSTMForecasterShardings,
        rngs: nnx.Rngs,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
    ):
        self.forward_block = ForecasterBlock(
            config=config,
            in_features=in_features,
            intermediate_features=intermediate_features,
            shardings=shardings,
            rngs=rngs,
            reverse=False,
            dtype=dtype,
            param_dtype=param_dtype,
        )

        self.reverse_block = ForecasterBlock(
            config=config,
            in_features=in_features,
            intermediate_features=intermediate_features,
            shardings=shardings,
            rngs=rngs,
            reverse=True,
            dtype=dtype,
            param_dtype=param_dtype,
        )

    def __call__(self, series: jax.Array):
        forward_seq, _ = self.forward_block(series)
        reverse_seq, output = self.reverse_block(series)
        final_seq = forward_seq + reverse_seq
        return final_seq, output


class LSTMForecaster(nnx.Module):
    """
    Stacked LSTM forecaster with optional bidirectional processing.

    Architecture:
    - Input projection
    - N blocks of [LSTM -> LayerNorm -> Attention -> Residual]
    - Optional bidirectional blocks (forward + reverse)
    - Output heads with pre-norm
    """

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
        self.bidirectional = config.bidirectional

        self.quantiles = config.quantiles
        self.num_quantiles = len(config.quantiles)

        # Input projection: metrics -> hidden_features
        self.up_proj = nnx.Linear(
            in_features=config.num_metrics,
            out_features=config.hidden_features,
            use_bias=False,
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

        # Build stacked blocks

        @nnx.vmap
        def _create_block(rngs: nnx.Rngs):
            return ForecasterBlock(
                config=config,
                in_features=config.hidden_features,
                intermediate_features=config.hidden_features * 2,
                shardings=shardings,
                rngs=rngs,
                reverse=False,
                dtype=dtype,
                param_dtype=param_dtype,
            )

        def _create_bidirectional_block(rngs: nnx.Rngs):
            return BidirectionalForecasterBlock(
                config=config,
                in_features=config.hidden_features,
                intermediate_features=config.hidden_features * 2,
                shardings=shardings,
                rngs=rngs,
                dtype=dtype,
                param_dtype=param_dtype,
            )

        block_rngs = rngs.fork(split=config.num_blocks)
        self.blocks = (
            _create_bidirectional_block(block_rngs)
            if config.bidirectional
            else _create_block(block_rngs)
        )

        # Output heads
        Linear = partial(
            nnx.Linear,
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

        self.head_norm = nnx.LayerNorm(
            num_features=config.hidden_features,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
        )

        self.point_head = Linear(
            in_features=config.hidden_features,
            out_features=config.num_metrics * config.horizon,
        )

        self.quantile_head = Linear(
            in_features=config.hidden_features,
            out_features=config.num_metrics * config.horizon * self.num_quantiles,
        )

    def encode(self, series: jax.Array) -> tuple[jax.Array, EncodedBelief]:
        """
        Encode time series through stacked LSTM blocks.

        Args:
            series: (batch, devices, metrics, time)

        Returns:
            EncodedBelief with final representations
        """
        chex.assert_rank(series, 4)

        series_t = rearrange(series, "b d m t -> b d t m")
        projected = self.up_proj(series_t)

        @nnx.scan(in_axes=(0, nnx.Carry), out_axes=(nnx.Carry, 0))
        def _block_scan(
            block: tp.Union[BidirectionalForecasterBlock, ForecasterBlock],
            carry: jax.Array,
        ):
            hidden_seq, output = block(carry)
            return hidden_seq, output

        last_hidden_seq, accumulated_output = _block_scan(self.blocks, projected)
        # (b, d, t, h) -> (b, d, last_timestep, h)
        last_hidden_state = last_hidden_seq[:, :, -1, :]

        # in this case consider the device mixing made with attention
        if self.use_device_mixer:
            # there is an additional axis for accumulation, we only care about the last block
            # (blocks, batch, devices, time, hidden) -> (last_block, batch, devices, last_timestep, hidden)
            last_hidden_state = accumulated_output.belief[-1, :, :, -1, :]

        return last_hidden_state, accumulated_output

    def __call__(self, series: jax.Array) -> ForecasterOutput:
        """
        Full forward pass: encode + decode to predictions.

        Args:
            series: (batch, devices, metrics, time)
            initial_carry: Optional initial LSTM state

        Returns:
            ForecasterOutput with point and quantile predictions
        """

        # here last_hidden_state is from the last carry array
        last_hidden_state, encoding = self.encode(series)
        normed_belief = self.head_norm(last_hidden_state)
        prediction = self.point_head(normed_belief)
        prediction = rearrange(prediction, "b d (m h) -> b d m h", h=self.horizon)

        quantiles = self.quantile_head(normed_belief)
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
