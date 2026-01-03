import streamlit as st
from datetime import datetime, timedelta, time
from tradestation_python import TradeStation
from pandas import DataFrame
from tradestation_python.types.enums import Unit
from typing import Tuple, List, Iterable
from utils.data import BarDirectionLabel, get_df_cranges


def configure_date_range(
    default_range: timedelta = timedelta(days=90),
) -> Tuple[datetime, datetime]:
    # Calculate default dates: 3 months ending at most recent weekday (not today)
    today = datetime.now().date()

    # Find the most recent weekday (Monday=0, Sunday=6)
    days_back = 1  # Start with yesterday
    most_recent_weekday = today - timedelta(days=days_back)
    while most_recent_weekday.weekday() > 4:  # If it's weekend (Sat=5, Sun=6)
        days_back += 1
        most_recent_weekday = today - timedelta(days=days_back)

    # Calculate start date (3 months before the end date)
    # Approximate 3 months as 90 days for simplicity
    default_start_date = most_recent_weekday - default_range
    default_end_date = most_recent_weekday

    left, right = st.columns(2)
    with left:
        start_date, end_date = (
            st.date_input("start date", value=default_start_date),
            st.date_input("end date", value=default_end_date),
        )
    with right:
        start_time, end_time = (
            st.time_input("start time", time(9, 30)),
            st.time_input("end time", time(16)),
        )
    firstdate = datetime.combine(start_date, start_time)
    lastdate = datetime.combine(end_date, end_time)
    return (firstdate, lastdate)


def configure_unit_interval() -> Tuple[Unit, int]:
    left, right = st.columns(2)
    with left:
        unit = st.selectbox(
            "Select Unit",
            options=[Unit.MINUTE, Unit.DAILY, Unit.WEEKLY, Unit.MONTHLY],
            index=1,
            format_func=lambda x: x.value,
        )
    with right:
        interval = st.number_input(
            "Interval", min_value=1, max_value=1440, value=1, step=1
        )
    return (unit, interval)


def configure_df_cranges(
    ts: TradeStation,
) -> Iterable[Tuple[DataFrame, List[Tuple[datetime, datetime, float, float]]]]:
    symbols = st.text_area("symbols", value="SMCI,PLTR,AMD,NVDA")
    firstdate, lastdate = configure_date_range(timedelta(days=5 * 365))
    unit, interval = configure_unit_interval()
    slope_tol = st.slider("slope tolerance", min_value=0.0, max_value=1.0, value=0.05)
    max_lookback = st.slider("max lookback", min_value=2, max_value=1000, value=14)

    df_cranges = get_df_cranges(
        _ts=ts,
        symbols=symbols,
        firstdate=firstdate,
        lastdate=lastdate,
        unit=unit,
        interval=interval,
        slope_tol=slope_tol,
        max_lookback=max_lookback,
    )
    return df_cranges


def configure_monte_carlo_params():
    num_steps = st.number_input("num steps", min_value=1, max_value=100, value=25)
    num_simulations = st.number_input(
        "num simulations", min_value=1, max_value=100000, value=10000
    )
    start_state = st.selectbox(
        "start state",
        options=[
            BarDirectionLabel.UP,
            BarDirectionLabel.DOWN,
            BarDirectionLabel.SIDEWAYS,
            BarDirectionLabel.OUT,
        ],
        index=0,
        format_func=lambda x: x.name,
    )
    return num_steps, num_simulations, start_state
