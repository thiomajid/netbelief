import typing as tp

import jax
import matplotlib.pyplot as plt
import numpy as np

COLOR_CONTEXT = "#2E86AB"
COLOR_FORECAST = "#E63946"
COLOR_GROUND_TRUTH = "#06A77D"
COLOR_CUTOFF_LINE = "#6C757D"
COLOR_ERROR = "#FFB703"


def plot_forecast(
    context: jax.Array | np.ndarray,
    forecasts: jax.Array | np.ndarray | None = None,
    quantile_forecasts: jax.Array | np.ndarray | None = None,
    quantiles: tp.Iterable[float] | None = None,
    ground_truth: jax.Array | np.ndarray | None = None,
    ax=None,
    title: str | None = None,
):
    if ax is None:
        ax = plt.gca()

    context = np.asarray(jax.device_get(context))
    if forecasts is not None:
        forecasts = np.asarray(jax.device_get(forecasts))
    if quantile_forecasts is not None:
        quantile_forecasts = np.asarray(jax.device_get(quantile_forecasts))
    if ground_truth is not None:
        ground_truth = np.asarray(jax.device_get(ground_truth))

    context_len = len(context)
    context_indices = np.arange(context_len)
    ax.plot(
        context_indices,
        context,
        color=COLOR_CONTEXT,
        label="Context",
        linewidth=2.5,
        marker="o",
        markersize=4,
        markevery=2,
    )

    # draw cutoff line at the last context point
    ax.axvline(
        x=context_len - 1,
        color=COLOR_CUTOFF_LINE,
        linestyle=":",
        linewidth=2,
        alpha=0.7,
        label="Forecast Start",
    )

    forecast_indices = None
    forecast_with_bridge = None

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
            label="Forecast (Point)",
            linewidth=3,
            marker="s",
            markersize=5,
            markevery=1,
        )

    if quantile_forecasts is not None:
        q_len = quantile_forecasts.shape[-1]
        if forecast_indices is None:
            forecast_indices = np.arange(context_len - 1, context_len - 1 + q_len + 1)

        num_q = quantile_forecasts.shape[0]
        cmap = plt.get_cmap("tab10")

        # Different line styles for better distinction
        line_styles = ["-", "--", "-.", ":"]
        markers = ["v", "^", "<", ">", "d", "p", "*", "h"]

        for i in range(num_q):
            q_val = quantile_forecasts[i]
            q_with_bridge = np.concatenate([[context[-1]], q_val])

            q_label = (
                f"Q-{list(quantiles)[i]}" if quantiles is not None else f"Q-idx{i}"
            )
            color = cmap(i % 10)
            linestyle = line_styles[i % len(line_styles)]
            marker = markers[i % len(markers)]

            ax.plot(
                forecast_indices,
                q_with_bridge,
                color=color,
                label=q_label,
                linewidth=2,
                alpha=0.8,
                linestyle=linestyle,
                marker=marker,
                markersize=4,
                markevery=2,
            )

            if forecast_with_bridge is not None:
                ax.fill_between(
                    forecast_indices,
                    forecast_with_bridge,
                    q_with_bridge,
                    color=color,
                    alpha=0.15,
                )

    if ground_truth is not None:
        if forecast_indices is not None:
            target_indices = forecast_indices
        else:
            target_indices = np.arange(
                context_len - 1, context_len - 1 + len(ground_truth) + 1
            )

        ground_truth_len = min(len(ground_truth), len(target_indices) - 1)
        ground_truth_indices = target_indices[: ground_truth_len + 1]

        # create extended ground truth array starting with last context value
        ground_truth_with_bridge = np.concatenate(
            [[context[-1]], ground_truth[:ground_truth_len]]
        )

        ax.plot(
            ground_truth_indices,
            ground_truth_with_bridge,
            color=COLOR_GROUND_TRUTH,
            label="Ground Truth",
            linewidth=3,
            linestyle="-",
            alpha=0.9,
            marker="D",
            markersize=5,
            markevery=1,
        )

        # highlight prediction error area using fill_between if point forecast exists
        if forecast_with_bridge is not None:
            ax.fill_between(
                ground_truth_indices,
                forecast_with_bridge[: ground_truth_len + 1],
                ground_truth_with_bridge,
                color=COLOR_ERROR,
                alpha=0.25,
                label="Prediction Error",
            )

    ax.set_xlabel("Time Step", fontsize=11, fontweight="bold")
    ax.set_ylabel("Value", fontsize=11, fontweight="bold")
    ax.legend(loc="best", framealpha=0.9, fontsize=9, ncol=2)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=15)

    return ax
