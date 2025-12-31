import typing as tp

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import initializers
from flax.nnx.nn import dtypes
from flax.typing import Axes
from jax import lax

from src.utils.types import RevINNormOutput


class RevIN(nnx.Module):
    def __init__(
        self,
        num_features: int,
        rngs: nnx.Rngs,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
        scale_init=initializers.ones_init(),
        bias_init=initializers.zeros_init(),
        reduction_axis: Axes = 1,
        axis_index_groups: tp.Any = None,
        epsilon: float = 1e-5,
    ):
        self.num_features = num_features
        self.reduction_axis = reduction_axis
        self.axis_index_groups = axis_index_groups
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.epsilon = epsilon
        self.promote_dtype = dtypes.promote_dtype

        feature_shape = (num_features,)
        self.scale = nnx.Param(scale_init(rngs.params(), feature_shape, param_dtype))
        self.bias = nnx.Param(bias_init(rngs.params(), feature_shape, param_dtype))

    def normalize(self, x: jax.Array):
        scale = self.scale[...]
        bias = self.bias[...]
        x, scale, bias = self.promote_dtype((x, scale, bias), dtype=self.dtype)

        mean = jnp.mean(x, axis=self.reduction_axis, keepdims=True)
        var = jnp.var(x, axis=self.reduction_axis, keepdims=True)
        inverse_std = lax.rsqrt(var + self.epsilon)

        x = (x - mean) * inverse_std
        normed = (x * scale) + bias

        sg_mean = lax.stop_gradient(mean)
        sg_inv_std = lax.stop_gradient(inverse_std)

        return RevINNormOutput(
            mean=sg_mean,
            inverse_std=sg_inv_std,
            normalized=normed,
        )

    def denormalize(self, output: RevINNormOutput):
        x = output.normalized

        scale, bias = self.scale.value, self.bias.value
        x, scale, bias = self.promote_dtype((x, scale, bias), dtype=self.dtype)

        x = (x - bias) / scale
        std = 1 / output.inverse_std
        x = x * std + output.mean

        return x

    def __call__(self, x: jax.Array):
        return self.normalize(x)
