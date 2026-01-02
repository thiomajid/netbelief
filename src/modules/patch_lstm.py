import typing as tp
from functools import partial

import chex
import jax
import jax.numpy as jnp
from einops import rearrange
from flax import nnx
from flax.nnx import initializers

from src.modules.attention import GroupedQueryAttention
from src.modules.lstm_config import LSTMForecasterShardings, PatchLSTMConfig
from src.modules.normalization import RevIN
from src.utils.types import EncodedBelief, ForecasterOutput


class Patcher(nnx.Module):
    def __init__(self, patch_size: int, stride: int, pad_mode: str = "edge"):
        self.patch_size = patch_size
        self.stride = stride
        self.pad_mode = pad_mode

    def create_patches(self, series: jax.Array):
        """
        Extract patches from time series (PatchTST-style) with optional padding.

        Args:
            series: jax.Array of shape (..., n_vars, seq_len)
            patch_size: int, length of each patch
            stride: int, step between patches
            pad_mode: str, padding mode ('edge', 'constant', etc.)

        Returns:
            patches: jax.Array of shape (..., n_vars, n_patches, patch_size)
        """

        chex.assert_rank(series, 4)
        seq_len = series.shape[-1]

        if seq_len < self.patch_size:
            # need to pad to at least patch_size
            pad_len = self.patch_size - seq_len
            # only pad time axis
            padding = [(0, 0)] * (series.ndim - 1) + [(0, pad_len)]
            series = jnp.pad(series, pad_width=padding, mode=self.pad_mode)
            seq_len = series.shape[-1]

        # compute required padding so that (seq_len - patch_size) is divisible by stride
        remainder = (seq_len - self.patch_size) % self.stride
        if remainder != 0:
            pad_len = self.stride - remainder
            padding = [(0, 0)] * (series.ndim - 1) + [(0, pad_len)]
            series = jnp.pad(series, pad_width=padding, mode=self.pad_mode)
            seq_len = series.shape[-1]

        n_patches = (seq_len - self.patch_size) // self.stride + 1
        start_indices = jnp.arange(n_patches) * self.stride
        patch_offsets = jnp.arange(self.patch_size)
        indices = start_indices[:, None] + patch_offsets  # (n_patches, patch_size)

        patches = series[..., indices]  # -> (..., n_vars, n_patches, patch_size)

        return patches

    def reverse(self, patched_series: jax.Array):
        series = rearrange(
            patched_series,
            "... np ps -> ... (np ps)",
            ps=self.patch_size,
        )

        return series


class PatchLSTMBlock(nnx.Module):
    """
    Single block of LSTM + optional attention mixer.
    Can process sequences forward or backward while preserving temporal order.
    """

    def __init__(
        self,
        config: PatchLSTMConfig,
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
        self.num_metrics = config.num_metrics

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

        self.mixer_norm: tp.Optional[nnx.RMSNorm]
        self.mixer: tp.Optional[GroupedQueryAttention]

        if config.use_device_mixer:
            self.mixer_norm = nnx.RMSNorm(
                num_features=intermediate_features,
                rngs=rngs,
                dtype=dtype,
                param_dtype=param_dtype,
                scale_init=nnx.with_partitioning(
                    initializers.ones_init(),
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
                implementation=config.attention_impl,
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

        self.down_proj = nnx.Linear(
            in_features=intermediate_features,
            out_features=in_features,
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

    def __call__(self, series: jax.Array) -> tuple[jax.Array, EncodedBelief]:
        """
        Process series through LSTM + attention.

        Args:
            series: (batch, devices, metrics, num_patches, in_features)

        Returns:
            Tuple of:
            - Transformed sequence (batch * devices * metrics, num_patches, out_features=in_features)
            - EncodedBelief with last hidden state info
        """
        chex.assert_rank(series, 5)

        batch, num_devices, num_metrics, num_patches, _ = series.shape
        series = rearrange(series, "b d m np h -> (b d m) np h")
        hidden_seq = self.rnn(series, reverse=self.reverse)  # (b, np, intermediate)
        last_hidden_state = hidden_seq[:, -1, :]

        if not self.use_device_mixer:
            out_seq = self.down_proj(hidden_seq)  # (b, np, in_features)
            out_seq = rearrange(
                out_seq,
                "(b d m) np h -> b d m np h",
                d=num_devices,
                m=num_metrics,
            )

            return out_seq, EncodedBelief(
                belief=last_hidden_state,
                rnn_hidden_seq=hidden_seq,
                rnn_last_hidden_state=last_hidden_state,
            )

        # learn device-to-device interactions
        hidden_seq = rearrange(
            hidden_seq,
            "(b d m) np h -> (b m np) d h",
            d=num_devices,
            m=num_metrics,
        )

        normed_hidden = self.mixer_norm(hidden_seq)
        mixer_output, attention_output = self.mixer(normed_hidden)
        belief = self.down_proj(mixer_output)
        belief = rearrange(
            belief,
            "(b m np) d h -> b d m np h",
            np=num_patches,
            m=self.num_metrics,
        )

        return belief, EncodedBelief(
            belief=belief,
            rnn_hidden_seq=hidden_seq,
            rnn_last_hidden_state=last_hidden_state,
            attention_ouput=attention_output,
        )


class PatchLSTM(nnx.Module):
    def __init__(
        self,
        config: PatchLSTMConfig,
        *,
        shardings: LSTMForecasterShardings,
        rngs: nnx.Rngs,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
    ):
        self.use_device_mixer = config.use_device_mixer
        self.patch_size = config.patch_size
        self.num_metrics = config.num_metrics
        self.horizon = config.horizon
        self.num_quantiles = len(config.quantiles)
        self.pooling_mode = config.pooling_mode

        self.revin = RevIN(
            config.num_metrics,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            reduction_axis=-1,
            scale_init=nnx.with_partitioning(
                initializers.ones_init(), sharding=shardings.norm
            ),
            bias_init=nnx.with_partitioning(
                initializers.zeros_init(),
                sharding=shardings.norm,
            ),
        )

        self.patcher = Patcher(
            patch_size=config.patch_size,
            stride=config.stride,
            pad_mode=config.pad_mode,
        )

        @nnx.vmap
        def _create_block(rngs: nnx.Rngs):
            return PatchLSTMBlock(
                config=config,
                in_features=config.hidden_features,
                intermediate_features=config.hidden_features * 2,
                shardings=shardings,
                rngs=rngs,
                reverse=False,
                dtype=dtype,
                param_dtype=param_dtype,
            )

        self.blocks = _create_block(rngs.fork(split=config.num_blocks))

        # output heads
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

        self.up_proj = Linear(
            in_features=config.patch_size,
            out_features=config.hidden_features,
        )

        self.head_norm = nnx.RMSNorm(
            num_features=config.hidden_features,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
        )

        self.point_head = Linear(
            in_features=config.hidden_features,
            out_features=config.horizon,
        )

        self.quantile_head = Linear(
            in_features=config.hidden_features,
            out_features=config.horizon * self.num_quantiles,
        )

    def encode(self, series: jax.Array):
        """
        Encode time series through stacked LSTM blocks.

        Args:
            series: (batch, devices, metrics, time)

        Returns:
            EncodedBelief with final representations
        """
        chex.assert_rank(series, 4)
        revin_out = self.revin(series)

        # patches is (b, d, m, np, ps)
        patches = self.patcher.create_patches(revin_out.normalized)
        patches = self.up_proj(patches)

        @nnx.scan(in_axes=(0, nnx.Carry), out_axes=(nnx.Carry, 0))
        def _block_scan(
            block: PatchLSTMBlock,
            carry: jax.Array,
        ):
            hidden_seq, output = block(carry)
            return hidden_seq, output

        # last_hidden_seq is (batch, devices, metrics, num_patches, hidden)
        last_hidden_seq, accumulated_output = _block_scan(self.blocks, patches)

        last_hidden_state: jax.Array
        if self.pooling_mode == "last":
            if self.use_device_mixer:
                # there is an additional axis for accumulation, we only care about the last block
                # (blocks, batch, devices, metrics, num_patches, hidden) -> (last_block, batch, devices, metrics, last_patch, hidden)
                last_hidden_state = accumulated_output.belief[-1, :, :, :, -1, :]
            else:
                last_hidden_state = last_hidden_seq[:, :, :, -1, :]
        elif self.pooling_mode == "mean":
            last_hidden_state = last_hidden_seq.mean(axis=-2)
        elif self.pooling_mode == "max":
            last_hidden_state = last_hidden_seq.max(axis=-2)

        # last_hidden_state is (b, d, m, hidden)
        return last_hidden_state, accumulated_output, revin_out

    def __call__(self, series: jax.Array):
        last_hidden_state, encoding, revin_out = self.encode(series)
        normed_belief = self.head_norm(last_hidden_state)
        # prediction (b, d, m, horizon)
        prediction = self.point_head(normed_belief)
        quantiles = self.quantile_head(normed_belief)
        quantiles = rearrange(
            quantiles,
            "b d m (h q) -> q b d m h",
            h=self.horizon,
            q=self.num_quantiles,
        )

        def reverse_revin(x: jax.Array):
            return self.revin.denormalize(revin_out.replace(normalized=x))

        unscaled_prediction = reverse_revin(prediction)
        unscaled_quantiles = jax.vmap(reverse_revin)(quantiles)

        return ForecasterOutput(
            encoding=encoding,
            point_predictions=unscaled_prediction,
            quantile_predictions=unscaled_quantiles,
        )
