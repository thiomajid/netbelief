import jax
import jax.numpy as jnp
import optax

from src.utils.types import ForecastingLossWithQuantiles


def root_mean_squared_error(predictions: jax.Array, targets: jax.Array):
    mse = optax.losses.squared_error(predictions, targets).mean()
    return jnp.sqrt(mse)


def mean_absolute_error(predictions: jax.Array, targets: jax.Array):
    error = jnp.abs(predictions - targets)
    return error.mean()


def pinball_loss(predictions: jax.Array, targets: jax.Array, quantile: jax.Array):
    """
    Compute quantile regression loss.

    Args:
        predictions: (batch, devices, metrics, horizon)
        targets: (batch, devices, metrics, horizon)
        quantile: scalar in [0, 1]
    """
    error = targets - predictions
    return jnp.mean(jnp.where(error >= 0, quantile * error, (quantile - 1) * error))


def multi_quantile_loss(
    quantile_predictions: jax.Array,
    targets: jax.Array,
    quantiles: jax.Array,
):
    """
    Returns a jax.Array containing the loss per quantile
    Args:
        quantile_predictions: (num_quantiles, batch, devices, horizon, metrics)
        targets: (batch, devices, horizon, metrics)
        quantiles: tuple of floats
    """

    losses = jax.vmap(pinball_loss, in_axes=(0, None, 0))(
        quantile_predictions, targets, quantiles
    )

    return jnp.array(losses)


def dual_head_loss(
    point_predictions: jax.Array,
    quantile_predictions: jax.Array,
    targets: jax.Array,
    quantiles: jax.Array,
    point_weight: float = 0.4,
):
    """Combined loss for point forecast + quantile forecasts"""

    # Point forecast loss (MSE - good for mean forecasts)
    point_loss = optax.losses.squared_error(
        predictions=point_predictions,
        targets=targets,
    ).mean()

    root_point_loss = jnp.sqrt(point_loss)

    quantile_losses = multi_quantile_loss(quantile_predictions, targets, quantiles)
    q_loss = jnp.mean(quantile_losses)

    total_loss = point_weight * root_point_loss + (1 - point_weight) * q_loss
    return ForecastingLossWithQuantiles(
        total=total_loss,
        mse=point_loss,
        rmse=root_point_loss,
        q_loss=q_loss,
        quantile_losses=quantile_losses,
        mae=mean_absolute_error(point_predictions, targets),
    )
