import typing as tp
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import initializers
from jax import lax

from src.modules.lstm import (
    LSTMForecaster,
    LSTMForecasterConfig,
    LSTMForecasterShardings,
)
from src.utils.types import PlannerOutput, ShardingRule


@dataclass
class MlpConfig:
    in_features: int
    hidden_features: int
    num_rules: int


@dataclass
class MlpShardings:
    in_kernel: ShardingRule = (None, "tp")
    out_kernel: ShardingRule = ("tp", None)


class MLP(nnx.Module):
    def __init__(
        self,
        config: MlpConfig,
        *,
        shardings: MlpShardings,
        rngs: nnx.Rngs,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
    ):
        Linear = partial(
            nnx.Linear,
            in_features=config.in_features,
            out_features=config.hidden_features * 4,
            use_bias=False,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=nnx.with_partitioning(
                initializers.lecun_normal(),
                sharding=shardings.in_kernel,
            ),
        )

        self.gate_proj = Linear()
        self.up_proj = Linear()
        self.down_proj = Linear(
            in_features=config.hidden_features * 4,
            out_features=config.num_rules,
            kernel_init=nnx.with_partitioning(
                initializers.lecun_normal(),
                sharding=shardings.out_kernel,
            ),
        )

    def __call__(self, belief: jax.Array):
        gate = jax.nn.swish(self.gate_proj(belief))
        up = self.up_proj(belief)
        hidden = gate * up
        action_logits = self.down_proj(hidden)
        return action_logits


@dataclass
class PlannerConfig:
    num_heads: int
    forecaster: LSTMForecasterConfig
    policy: MlpConfig
    attention_bias: bool = False
    normalize_qk: bool = True


@dataclass
class PlannerShardings:
    forecaster: LSTMForecasterShardings
    policy: MlpShardings
    attn_qkv_proj: ShardingRule = (None, "tp")
    attn_o_proj: ShardingRule = ("tp", None)


class LSTMPlanner(nnx.Module):
    def __init__(
        self,
        config: PlannerConfig,
        *,
        shardings: PlannerShardings,
        rngs: nnx.Rngs,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
    ):
        assert config.forecaster.hidden_features == config.policy.hidden_features, (
            "The forecasting module must have the same hidden dimension as the policy MLP"
        )

        self.forecaster = LSTMForecaster(
            config=config.forecaster,
            shardings=shardings.forecaster,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
        )

        hidden_dim = config.forecaster.hidden_features
        self.mixer = nnx.MultiHeadAttention(
            num_heads=config.num_heads,
            in_features=hidden_dim,
            out_features=hidden_dim,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=config.attention_bias,
            decode=False,
            normalize_qk=config.normalize_qk,
            kernel_init=nnx.with_partitioning(
                initializers.lecun_normal(),
                sharding=shardings.attn_qkv_proj,
            ),
            out_kernel_init=nnx.with_partitioning(
                initializers.lecun_normal(),
                sharding=shardings.attn_o_proj,
            ),
        )

        self.policy = MLP(
            config=config.policy,
            shardings=shardings,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
        )

    def __call__(
        self,
        series: jax.Array,
        initial_carry: tp.Optional[jax.Array] = None,
    ):
        sg_series = lax.stop_gradient(series)
        sg_carry = (
            lax.stop_gradient(initial_carry) if initial_carry is not None else None
        )

        # (batch, devices, hidden)
        belief = self.forecaster.encode(sg_series, sg_carry)
        weights = self.attention(belief)
        belief = belief + weights
        action_logits = self.policy(belief)

        return PlannerOutput(
            action_logits=action_logits,
            scores=jax.nn.sigmoid(action_logits),
        )
