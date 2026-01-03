import streamlit as st
from typing import Dict, List, Tuple
from datetime import datetime
from utils.data import convert_hex_to_rgba
from pandas import DataFrame, date_range
import numpy as np
from plotly.graph_objects import Figure, Scatter


def configure_fig() -> Dict[str, str]:
    volume_hex = st.color_picker("volume", "#84AAB7")
    volume_rgba = convert_hex_to_rgba(volume_hex, 0.4)

    return {"volume": volume_rgba}


def configure_plotly() -> dict:
    return {
        "scrollZoom": True,
        "displayModeBar": True,
        "displaylo": False,
        "modeBarButtonsToRemove": ["zoom", "select2d", "lasso2d"],
    }


def add_trendline_to_fig(
    fig: Figure,
    df: DataFrame,
    intercept: float,
    slope: float,
    num_smooth_points: int = 100,
) -> Figure:
    x_smooth = np.linspace(0, len(df) - 1, num_smooth_points)
    y_smooth = intercept + slope * x_smooth
    time_smooth = date_range(df.index[0], df.index[-1], num_smooth_points)

    # Add the trendline to the figure
    fig.add_trace(
        Scatter(
            x=time_smooth,
            y=y_smooth,
            mode="lines",
            line=dict(color="orange", width=2),
            name=f"{len(df)}-Period Trendline",
        ),
        secondary_y=True,
    )

    return fig


def add_volume_deltas_to_fig(
    fig: Figure,
    deltas: List[Tuple[datetime, float, float]],
    positive_color: str = "rgba(0, 255, 0, 0.6)",
    negative_color: str = "rgba(255, 0, 0, 0.6)",
) -> Figure:
    """
    Add volume deltas as circles to a Plotly figure.

    The circles are positioned at the midpoint of each bar and have diameters
    proportional to the volume delta magnitude.

    Args:
        fig: Plotly figure to add volume deltas to
        df: DataFrame with OHLC and volume data
        positive_color: Color for positive volume deltas (net buying pressure)
        negative_color: Color for negative volume deltas (net selling pressure)
        size_multiplier: Multiplier to scale circle sizes
        min_size: Minimum circle size to ensure visibility

    Returns:
        Uted Plotly figure with volume delta circles
    """
    if not deltas:
        return fig

    # Separate positive and negative deltas for different colors
    pos_dates, pos_midpoints, pos_sizes = [], [], []
    neg_dates, neg_midpoints, neg_sizes = [], [], []

    for date, midpoint, delta in deltas:
        if delta >= 0:
            pos_dates.append(date)
            pos_midpoints.append(midpoint)
            pos_sizes.append(abs(delta))
        else:
            neg_dates.append(date)
            neg_midpoints.append(midpoint)
            neg_sizes.append(abs(delta))

    # Add positive volume delta circles (net buying pressure)
    if pos_dates:
        fig.add_trace(
            Scatter(
                x=pos_dates,
                y=pos_midpoints,
                mode="markers",
                marker=dict(
                    size=pos_sizes,
                    color=positive_color,
                    line=dict(width=1, color="rgba(0, 0, 0, 0.3)"),
                    symbol="circle",
                ),
                name="Volume Delta (+)",
                hovertemplate="<b>Positive Volume Delta</b><br>"
                + "Date: %{x}<br>"
                + "Price: %{y:.2f}<br>"
                + "Net Buying Pressure<br>"
                + "<extra></extra>",
                showlegend=True,
            ),
            secondary_y=True,
        )

    # Add negative volume delta circles (net selling pressure)
    if neg_dates:
        fig.add_trace(
            Scatter(
                x=neg_dates,
                y=neg_midpoints,
                mode="markers",
                marker=dict(
                    size=neg_sizes,
                    color=negative_color,
                    line=dict(width=1, color="rgba(0, 0, 0, 0.3)"),
                    symbol="circle",
                ),
                name="Volume Delta (-)",
                hovertemplate="<b>Negative Volume Delta</b><br>"
                + "Date: %{x}<br>"
                + "Price: %{y:.2f}<br>"
                + "Net Selling Pressure<br>"
                + "<extra></extra>",
                showlegend=True,
            ),
            secondary_y=True,
        )

    return fig


def add_consolidation_ranges_to_fig(
    fig: Figure,
    ranges: List[Tuple[datetime, datetime, float, float]],
    high_color: str = "rgba(0, 255, 0, 0.3)",
    low_color: str = "rgba(255, 0, 0, 0.3)",
    line_width: int = 2,
) -> Figure:
    """
    Add horizontal lines representing consolidation ranges to a Plotly figure.

    Args:
        fig: Plotly figure to add ranges to
        ranges: List of tuples containing (start_date, end_date, high_value, low_value)
        high_color: Color for the high horizontal lines
        low_color: Color for the low horizontal lines
        line_width: Width of the horizontal lines

    Returns:
        Uted Plotly figure with consolidation ranges
    """
    for i, (start_date, end_date, high_val, low_val) in enumerate(ranges):
        # Add high horizontal line
        fig.add_trace(
            Scatter(
                x=[start_date, end_date],
                y=[high_val, high_val],
                mode="lines",
                line=dict(color=high_color, width=line_width, dash="dash"),
                name=f"Range {i + 1} High ({high_val:.2f})",
                showlegend=False,
                hovertemplate=f"High: {high_val:.2f}<br>Period: {start_date} to {end_date}<extra></extra>",
            ),
            secondary_y=True,
        )

        # Add low horizontal line
        fig.add_trace(
            Scatter(
                x=[start_date, end_date],
                y=[low_val, low_val],
                mode="lines",
                line=dict(color=low_color, width=line_width, dash="dash"),
                name=f"Range {i + 1} Low ({low_val:.2f})",
                showlegend=False,
                hovertemplate=f"Low: {low_val:.2f}<br>Period: {start_date} to {end_date}<extra></extra>",
            ),
            secondary_y=True,
        )

        # Optional: Add a filled area between high and low
        fig.add_trace(
            Scatter(
                x=[start_date, end_date, end_date, start_date, start_date],
                y=[low_val, low_val, high_val, high_val, low_val],
                fill="toself",
                fillcolor="rgba(128, 128, 128, 0.1)",
                line=dict(color="rgba(0,0,0,0)"),
                name=f"Range {i + 1} Area",
                showlegend=False,
                hovertemplate=f"Range: {low_val:.2f} - {high_val:.2f}<br>Period: {start_date} to {end_date}<extra></extra>",
            ),
            secondary_y=True,
        )

    return fig
