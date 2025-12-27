import typing as tp
from functools import partial

import chex
import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import initializers

from src.utils.types import AttentionImpl


class GroupedQueryAttention(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int,
        num_kv_heads: int,
        rngs: nnx.Rngs,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
        normalize_qk: bool = True,
        is_causal: bool = False,
        qkv_kernel_init=initializers.lecun_normal(),
        out_kernel_init=initializers.lecun_normal(),
        scale_init=initializers.ones_init(),
        implementation: AttentionImpl = "xla",
        local_window_size: tp.Optional[tp.Union[int, tuple[int, int]]] = None,
    ):
        chex.assert_equal(in_features % num_heads, 0)

        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.head_dim = in_features // num_heads
        self.normalize_qk = normalize_qk
        self.is_causal = is_causal
        self.implementation = implementation
        self.local_window_size = local_window_size

        Linear = partial(
            nnx.LinearGeneral,
            in_features=in_features,
            use_bias=False,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
        )

        self.q_proj = Linear(
            out_features=(num_heads, self.head_dim),
            kernel_init=qkv_kernel_init,
        )

        self.k_proj = Linear(
            out_features=(num_kv_heads, self.head_dim),
            kernel_init=qkv_kernel_init,
        )

        self.v_proj = Linear(
            out_features=(num_kv_heads, self.head_dim),
            kernel_init=qkv_kernel_init,
        )

        self.o_proj = Linear(
            in_features=(num_heads, self.head_dim),
            out_features=out_features,
            axis=(-2, -1),
            kernel_init=out_kernel_init,
        )

        self.q_norm: tp.Optional[nnx.RMSNorm]
        self.k_norm: tp.Optional[nnx.RMSNorm]

        if normalize_qk:
            self.q_norm = nnx.RMSNorm(
                self.head_dim,
                rngs=rngs,
                dtype=dtype,
                param_dtype=param_dtype,
                scale_init=scale_init,
            )

            self.k_norm = nnx.RMSNorm(
                self.head_dim,
                rngs=rngs,
                dtype=dtype,
                param_dtype=param_dtype,
                scale_init=scale_init,
            )
        else:
            self.q_norm = nnx.data(None)
            self.k_norm = nnx.data(None)

    def __call__(
        self,
        x: jax.Array,
        mask: tp.Optional[jax.Array] = None,
    ):
        batch, num_devices, _ = x.shape
        query, key, value = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        if self.normalize_qk:
            query = self.q_norm(query)
            key = self.k_norm(key)

        attention = jax.nn.dot_product_attention(
            query=query,
            key=key,
            value=value,
            mask=mask,
            is_causal=self.is_causal,
            implementation=self.implementation,
            local_window_size=self.local_window_size,
        )

        output = self.o_proj(attention)
        return output, attention
