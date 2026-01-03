from typing import List, Tuple
import streamlit as st
from datetime import datetime
from tradestation_python import TradeStation
from plotly.graph_objects import Figure
from pandas import DataFrame
from utils.data import (
    convert_bars_to_df,
    convert_df_to_fig,
    get_bars,
    consolidation_ranges,
    channel_fit,
)
from utils.plot import (
    configure_fig,
    configure_plotly,
    add_trendline_to_fig,
    add_volume_deltas_to_fig,
    add_consolidation_ranges_to_fig,
)
from utils.data import volume_deltas
from utils.configure import configure_date_range, configure_unit_interval


def configure_df(ts: TradeStation) -> DataFrame:
    # input a symbol
    st.session_state.symbol = st.text_input(label="symbol", value="SMCI")

    # select date range
    firstdate, lastdate = configure_date_range()
    unit, interval = configure_unit_interval()

    # fetch bars using the time range
    try:
        bars = get_bars(
            ts, st.session_state.symbol, firstdate, lastdate, unit, interval
        )
    except Exception as e:
        st.error(f"Error fetching bars: {e}")
        get_bars.clear()
        st.stop()

    # convert bars to dataframe
    if not bars:
        st.stop()
    df = convert_bars_to_df(bars)

    return df


def configure_trend_df(df: DataFrame) -> DataFrame:
    left, right = st.columns([4, 1])
    with left:
        date = st.select_slider("start trend", df.index)
    with right:
        periods = st.number_input(
            "lookback",
            min_value=2,
            max_value=len(df) // 2,
            value=5,
        )
    # slice df from the start date ng back the number of periods
    trend_df = df[df.index <= date].tail(periods)

    return trend_df


def add_channel_to_fig(fig: Figure, df: DataFrame) -> Figure:
    h_slope, h_intercept, l_slope, l_intercept = channel_fit(df)
    fig = add_trendline_to_fig(fig, df, h_intercept, h_slope)
    fig = add_trendline_to_fig(fig, df, l_intercept, l_slope)
    return fig


def configure_consolidation_ranges(
    df: DataFrame,
) -> List[Tuple[datetime, datetime, float, float]]:
    left, right = st.columns(2)
    with left:
        slope_tol = st.number_input("slope tolerance", value=0.05)
    with right:
        max_lookback = st.number_input("max lookback", value=10)
    cranges = consolidation_ranges(df, slope_tol, max_lookback)
    return cranges


def configure_volume_deltas(df: DataFrame) -> List[Tuple[datetime, float, float]]:
    scale = st.slider(
        "volume delta scale", min_value=0.1, max_value=1000.0, value=100.0
    )
    return volume_deltas(df, scale)


if __name__ == "__main__":
    ts = TradeStation()

    with st.sidebar:
        df = configure_df(ts)
        fconfig = configure_fig()
        pconfig = configure_plotly()
        cranges = configure_consolidation_ranges(df)
        vdeltas = configure_volume_deltas(df)

    fig = convert_df_to_fig(df, fconfig)
    fig = add_consolidation_ranges_to_fig(fig, cranges)
    fig = add_volume_deltas_to_fig(fig, vdeltas)
    st.plotly_chart(fig, width="stretch", config=pconfig)
