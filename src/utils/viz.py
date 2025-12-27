import jax
import matplotlib.pyplot as plt
import numpy as np

COLOR_CONTEXT = "#4a90d0"
COLOR_FORECAST = "#d94e4e"
COLOR_GROUND_TRUTH = "#4a90d0"
COLOR_CUTOFF_LINE = "#000000"
COLOR_ERROR = "#D9B99B"


def plot_forecast(
    context: jax.Array | np.ndarray,
    forecasts: jax.Array | np.ndarray | None = None,
    ground_truth: jax.Array | np.ndarray | None = None,
    ax=None,
    title: str | None = None,
):
    if ax is None:
        ax = plt.gca()

    context = np.asarray(jax.device_get(context))
    if forecasts is not None:
        forecasts = np.asarray(jax.device_get(forecasts))
    if ground_truth is not None:
        ground_truth = np.asarray(jax.device_get(ground_truth))

    context_len = len(context)
    context_indices = np.arange(context_len)
    ax.plot(context_indices, context, color=COLOR_CONTEXT, label="Context", linewidth=2)

    # draw cutoff line at the last context point
    ax.axvline(
        x=context_len - 1,
        color=COLOR_CUTOFF_LINE,
        linestyle="--",
        linewidth=1,
        alpha=0.5,
    )

    if forecasts is not None:
        forecast_len = len(forecasts)
        # start forecast indices from the last context point for continuity
        # add 1 to length to include the bridge point
        forecast_indices = np.arange(
            context_len - 1, context_len - 1 + forecast_len + 1
        )

        # create extended forecast array starting with last context value
        forecast_with_bridge = np.concatenate([[context[-1]], forecasts])
        ax.plot(
            forecast_indices,
            forecast_with_bridge,
            color=COLOR_FORECAST,
            label="Forecast",
            linewidth=2,
        )

        if ground_truth is not None:
            ground_truth_len = min(len(ground_truth), forecast_len)
            ground_truth_indices = forecast_indices[: ground_truth_len + 1]

            # create extended ground truth array starting with last context value
            ground_truth_with_bridge = np.concatenate(
                [[context[-1]], ground_truth[:ground_truth_len]]
            )

            ax.plot(
                ground_truth_indices,
                ground_truth_with_bridge,
                color=COLOR_GROUND_TRUTH,
                label="Ground Truth",
                linewidth=2,
                linestyle=":",
                alpha=0.8,
            )

            # highlight prediction error area using fill_between
            # include the bridge point to start error shading at the cutoff line
            ax.fill_between(
                ground_truth_indices,
                forecast_with_bridge[: ground_truth_len + 1],
                ground_truth_with_bridge,
                color=COLOR_ERROR,
                alpha=0.3,
                label="Prediction Error",
            )

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    if title:
        ax.set_title(title)

    return ax
