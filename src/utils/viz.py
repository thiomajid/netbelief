import typing as tp

import jax
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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
    fig: go.Figure | None = None,
    title: str | None = None,
):
    """
    Plots time series forecast using Plotly.

    Args:
        context: Historical data points.
        forecasts: Point forecasts.
        quantile_forecasts: Quantile forecasts (num_quantiles, forecast_len).
        quantiles: List of quantile values.
        ground_truth: Actual values for the forecast period.
        fig: Existing Plotly figure to add to.
        title: Plot title.
    """
    if fig is None:
        fig = go.Figure()

    context = np.asarray(jax.device_get(context))
    if forecasts is not None:
        forecasts = np.asarray(jax.device_get(forecasts))
    if quantile_forecasts is not None:
        quantile_forecasts = np.asarray(jax.device_get(quantile_forecasts))
    if ground_truth is not None:
        ground_truth = np.asarray(jax.device_get(ground_truth))

    context_len = len(context)
    context_indices = np.arange(context_len)

    # Plot Context
    fig.add_trace(
        go.Scatter(
            x=context_indices,
            y=context,
            mode="lines+markers",
            name="Context",
            line=dict(color=COLOR_CONTEXT, width=2.5),
            marker=dict(size=4),
        )
    )

    # Draw cutoff line
    fig.add_vline(
        x=context_len - 1,
        line_width=2,
        line_dash="dot",
        line_color=COLOR_CUTOFF_LINE,
        annotation_text="Forecast Start",
        annotation_position="top left",
    )

    forecast_indices = None
    forecast_with_bridge = None

    if forecasts is not None:
        forecast_len = len(forecasts)
        forecast_indices = np.arange(
            context_len - 1, context_len - 1 + forecast_len + 1
        )
        forecast_with_bridge = np.concatenate([[context[-1]], forecasts])

        fig.add_trace(
            go.Scatter(
                x=forecast_indices,
                y=forecast_with_bridge,
                mode="lines+markers",
                name="Forecast (Point)",
                line=dict(color=COLOR_FORECAST, width=3),
                marker=dict(symbol="square", size=5),
            )
        )

    if quantile_forecasts is not None:
        q_len = quantile_forecasts.shape[-1]
        if forecast_indices is None:
            forecast_indices = np.arange(context_len - 1, context_len - 1 + q_len + 1)

        num_q = quantile_forecasts.shape[0]
        colors = px.colors.qualitative.Plotly

        for i in range(num_q):
            q_val = quantile_forecasts[i]
            q_with_bridge = np.concatenate([[context[-1]], q_val])
            q_label = (
                f"Q-{list(quantiles)[i]}" if quantiles is not None else f"Q-idx{i}"
            )
            color = colors[i % len(colors)]

            fig.add_trace(
                go.Scatter(
                    x=forecast_indices,
                    y=q_with_bridge,
                    mode="lines+markers",
                    name=q_label,
                    line=dict(color=color, width=2, dash="dash"),
                    marker=dict(size=4),
                    opacity=0.8,
                )
            )

            if forecast_with_bridge is not None:
                # Fill between forecast and quantile
                fig.add_trace(
                    go.Scatter(
                        x=np.concatenate([forecast_indices, forecast_indices[::-1]]),
                        y=np.concatenate([forecast_with_bridge, q_with_bridge[::-1]]),
                        fill="toself",
                        fillcolor=color,
                        opacity=0.15,
                        line=dict(color="rgba(255,255,255,0)"),
                        hoverinfo="skip",
                        showlegend=False,
                        name=f"Area {q_label}",
                    )
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
        ground_truth_with_bridge = np.concatenate(
            [[context[-1]], ground_truth[:ground_truth_len]]
        )

        fig.add_trace(
            go.Scatter(
                x=ground_truth_indices,
                y=ground_truth_with_bridge,
                mode="lines+markers",
                name="Ground Truth",
                line=dict(color=COLOR_GROUND_TRUTH, width=3),
                marker=dict(symbol="diamond", size=5),
                opacity=0.9,
            )
        )

        if forecast_with_bridge is not None:
            # Prediction Error Area
            y_upper = forecast_with_bridge[: ground_truth_len + 1]
            y_lower = ground_truth_with_bridge

            fig.add_trace(
                go.Scatter(
                    x=np.concatenate(
                        [ground_truth_indices, ground_truth_indices[::-1]]
                    ),
                    y=np.concatenate([y_upper, y_lower[::-1]]),
                    fill="toself",
                    fillcolor=COLOR_ERROR,
                    opacity=0.25,
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    showlegend=True,
                    name="Prediction Error",
                )
            )

    fig.update_layout(
        title=title if title else "",
        xaxis_title="Time Step",
        yaxis_title="Value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        hovermode="x unified",
    )

    return fig
