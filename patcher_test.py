import jax
import jax.numpy as jnp
from flax import nnx, struct
from jax.sharding import AxisType

from src.modules.lstm import (
    LSTMForecasterShardings,
)
from src.modules.lstm_config import PatchLSTMConfig
from src.modules.patch_lstm import PatchLSTM
from src.training.loss import multi_quantile_loss
from src.utils.types import ForecasterOutput

if __name__ == "__main__":
    config = PatchLSTMConfig(
        hidden_features=128,
        num_metrics=10,
        head_bias=True,
        horizon=10,
        num_heads=4,
        use_device_mixer=True,
        patch_size=16,
        stride=16,
    )

    shardings = LSTMForecasterShardings()
    mesh = jax.make_mesh(
        (1, 1),
        ("dp", "tp"),
        axis_types=(AxisType.Auto, AxisType.Auto),
    )

    rngs = nnx.Rngs(123)
    batch = 5
    num_devices = 6
    timesteps = 64

    input = jax.random.normal(
        rngs(),
        shape=(batch, num_devices, config.num_metrics, timesteps),
    )

    target = jax.random.normal(
        rngs(),
        shape=(batch, num_devices, config.num_metrics, config.horizon),
    )

    run = nnx.jit(lambda m, x: m(x))

    jax.set_mesh(mesh)

    model = PatchLSTM(
        config=config,
        shardings=shardings,
        rngs=rngs,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
    )

    output: ForecasterOutput = run(model, input)

    print(output.point_predictions.shape)
    print(output.quantile_predictions.shape)

    quantiles = jnp.asarray(config.quantiles)

    print(multi_quantile_loss(output.quantile_predictions, target, quantiles))
    print(struct.serialization.to_state_dict(target=output).keys())
