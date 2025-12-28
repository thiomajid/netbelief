import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import initializers

from src.modules.lstm_config import LSTMForecasterConfig, LSTMForecasterShardings


class CausalDepthwiseSeparableConvolution(nnx.Module):
    def __init__(
        self,
        config: LSTMForecasterConfig,
        in_features: int,
        out_features: int,
        *,
        shardings: LSTMForecasterShardings,
        rngs: nnx.Rngs,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
    ):
        self.depthwise = nnx.Conv(
            in_features=in_features,
            out_features=in_features,
            kernel_size=(config.conv_kernel_size,),
            strides=(1,),
            feature_group_count=in_features,  # one filter per input channel
            padding="CAUSAL",
            use_bias=config.conv_bias,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=nnx.with_partitioning(
                initializers.lecun_normal(),
                sharding=shardings.proj_conv_kernel,
            ),
            bias_init=nnx.with_partitioning(
                initializers.zeros_init(),
                sharding=shardings.proj_conv_bias,
            ),
        )

        # mix channels at each timestep
        self.pointwise = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=(1,),
            strides=(1,),
            padding="VALID",
            use_bias=config.conv_bias,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=nnx.with_partitioning(
                initializers.lecun_normal(),
                sharding=shardings.proj_conv_kernel,
            ),
            bias_init=nnx.with_partitioning(
                initializers.zeros_init(),
                sharding=shardings.proj_conv_bias,
            ),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Apply depthwise separable convolution.

        Args:
            x: (batch, time, features) or (batch, devices, time, features)

        Returns:
            Transformed array with same time dimension
        """
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
