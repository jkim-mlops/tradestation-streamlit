from collections import Counter, defaultdict
from enum import Enum
from pandas import DataFrame, Series, Timestamp
from typing import Dict, Iterable, List, Tuple
from plotly.graph_objects import Bar, Figure, Candlestick
from plotly.subplots import make_subplots
import streamlit as st
from tradestation_python import TradeStation
from tradestation_python.types.enums import Unit
from datetime import datetime
import numpy as np
from sklearn.linear_model import LinearRegression


@st.cache_data()
def get_bars(
    _ts: TradeStation,
    symbol: str,
    firstdate: datetime,
    lastdate: datetime,
    unit: Unit = Unit.DAILY,
    interval: int = 1,
) -> list:
    return _ts.market_data.bars(
        symbol, firstdate=firstdate, lastdate=lastdate, unit=unit, interval=interval
    ).bars


def convert_bars_to_df(bars: list) -> DataFrame:
    """
    Convert a list of bar objects to a pandas DataFrame.

    Args:
        bars: List of bar objects with OHLC data and metadata

    Returns:
        DataFrame: DataFrame with bar data indexed by timestamp
    """
    if not bars:
        return DataFrame()

    # Extract data from each bar
    data = []
    for bar in bars:
        row = {
            "timestamp": bar.timestamp,
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "total_volume": int(bar.total_volume),
            "down_ticks": int(bar.down_ticks),
            "down_volume": int(bar.down_volume),
            "up_ticks": int(bar.up_ticks),
            "up_volume": int(bar.up_volume),
            "unchanged_ticks": int(bar.unchanged_ticks),
            "unchanged_volume": int(bar.unchanged_volume),
            "total_ticks": int(bar.total_ticks),
            "open_interest": int(bar.open_interest),
            "epoch": int(bar.epoch),
            "bar_status": bar.bar_status,
            "is_realtime": bar.is_realtime,
            "is_end_of_history": bar.is_end_of_history,
        }
        data.append(row)

    # Create DataFrame and set timestamp as index
    df = DataFrame(data)
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)

    return df


def get_dfs(
    ts: TradeStation,
    symbols: List[str],
    firstdate: datetime,
    lastdate: datetime,
    unit: Unit = Unit.DAILY,
    interval: int = 1,
) -> Iterable[DataFrame]:
    for symbol in symbols:
        bars = get_bars(ts, symbol, firstdate, lastdate, unit, interval)
        df = convert_bars_to_df(bars)
        yield df


def convert_hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    rgb = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})"


def convert_df_to_fig(df: DataFrame, config: dict[str, str]) -> Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC",
        ),
        secondary_y=True,
    )

    fig.add_trace(
        Bar(
            x=df.index,
            y=df["total_volume"],
            name="Total Volume",
            marker={"color": config["volume"]},
        ),
        secondary_y=False,
    )

    fig.update_layout(
        title=st.session_state.get("symbol", "Symbol"),
        yaxis_title="Volume",
        yaxis2_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=800,
    )

    return fig


def volume_deltas(
    df: DataFrame, scale: float = 100.0
) -> List[Tuple[datetime, float, float]]:
    rows = list(df.iterrows())

    dates = []
    midpoints = []
    up_vols_per_tick = []
    down_vols_per_tick = []
    max_vol_per_tick = 0

    for i in range(len(df)):
        date, series = rows[i]
        dates.append(Timestamp(str(date)).to_pydatetime())
        midpoints.append(float(series.low + (series.high - series.low) / 2))
        up_vol_per_tick = float(series.up_volume / series.up_ticks)
        down_vol_per_tick = float(series.down_volume / series.down_ticks)
        max_vol_per_tick = max([max_vol_per_tick, up_vol_per_tick, down_vol_per_tick])
        up_vols_per_tick.append(up_vol_per_tick)
        down_vols_per_tick.append(down_vol_per_tick)

    norm_vol_deltas = [
        (up - down) / max_vol_per_tick * scale
        for up, down in zip(up_vols_per_tick, down_vols_per_tick)
    ]

    return list(zip(dates, midpoints, norm_vol_deltas))


def calculate_rolling_trendline_slopes(
    df: DataFrame, n_periods: int, price_column: str = "close"
) -> Tuple[List[float], List[float], List[float]]:
    """
    Calculate rolling trendline slopes over n periods.

    Args:
        df: DataFrame with OHLC data
        n_periods: Number of periods to look back for trendline calculation
        price_column: Column to use for slope calculation ('close', 'high', 'low', etc.)

    Returns:
        Tuple of (slopes, r_squared, trendline_values)
        - slopes: Series of slope values for each period
        - r_squared: Series of R-squared values indicating trend strength
        - trendline_values: Series of trendline values at each period
    """
    slopes = []
    r_squared_values = []
    trendline_values = []

    for i in range(len(df)):
        if i < n_periods - 1:
            # Not enough data yet
            slopes.append(np.nan)
            r_squared_values.append(np.nan)
            trendline_values.append(np.nan)
        else:
            # Get the last n_periods of data
            window_data = df.iloc[i - n_periods + 1 : i + 1]

            # Create x values (0, 1, 2, ..., n_periods-1)
            x = np.arange(n_periods).reshape(-1, 1)
            y = window_data[price_column].values.astype(float)

            # Fit linear regression
            reg = LinearRegression().fit(x, y)  # type: ignore
            slope = reg.coef_[0]
            r_squared = reg.score(x, y)  # type: ignore

            # Calculate current trendline value (at the latest point)
            current_trendline_value = reg.predict(np.array([[n_periods - 1]]))[0]

            slopes.append(slope)
            r_squared_values.append(r_squared)
            trendline_values.append(current_trendline_value)

    return (slopes, r_squared_values, trendline_values)


def linear_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Perform linear regression and return slope and intercept.

    Args:
        x: Independent variable values
        y: Dependent variable values
    Returns:
        Tuple of (slope, intercept)
    """
    reg = LinearRegression().fit(x.reshape(-1, 1), y)
    return float(reg.coef_[0]), float(reg.intercept_)


def calculate_rolling_trendline_slopes_before_date(
    df: DataFrame, n_periods: int, cutoff_date: str, price_column: str = "close"
) -> Tuple[List[float], List[float], List[float]]:
    """
    Calculate rolling trendline slopes for data before a specific date.
    """
    # Filter DataFrame to only include dates before cutoff_date
    filtered_df = df[df.index < cutoff_date]

    # Calculate slopes on filtered data
    return calculate_rolling_trendline_slopes(filtered_df, n_periods, price_column)


def calculate_slope_change(slopes: Series, change_periods: int = 1) -> Series:
    """
    Calculate the change in slope over specified periods.

    Args:
        slopes: Series of slope values
        change_periods: Number of periods to look back for slope change calculation

    Returns:
        Series of slope changes
    """
    return slopes.diff(change_periods)


def consolidation_ranges(
    df: DataFrame, slope_tol: float = 0.05, max_lookback: int = 10
) -> List[Tuple[datetime, datetime, float, float]]:
    rows = list(df.iterrows())
    res = []
    for curr_idx in range(len(rows)):
        # parse curr row
        crow = rows[curr_idx]
        curr_date, _ = crow
        lookback_idx = curr_idx

        # track high and low
        lookback_highs, lookback_lows = [], []
        lookback_highs_len, lookback_lows_len = 0, 0
        lookback_max, lookback_min = None, None

        while lookback_idx >= 0 and curr_idx - lookback_idx < max_lookback:
            # parse lookback row
            lrow = rows[lookback_idx]
            ldate, lseries = lrow

            # add vals to lookback
            lookback_highs.append(lseries.high)
            lookback_lows.append(lseries.low)

            # update lengths
            lookback_highs_len += 1
            lookback_lows_len += 1

            # update max and min
            if not lookback_max or lseries.high > lookback_max:
                lookback_max = lseries.high
            if not lookback_min or lseries.low < lookback_min:
                lookback_min = lseries.low

            # fit linreg and add to result if tolerance is met
            if lookback_highs_len >= 2:
                high_slope, _ = linear_regression(
                    np.array(range(lookback_highs_len)), np.array(lookback_highs)
                )
                low_slope, _ = linear_regression(
                    np.array(range(lookback_lows_len)), np.array(lookback_lows)
                )
                if abs(high_slope) < slope_tol and abs(low_slope) < slope_tol:
                    res.append((ldate, curr_date, lookback_max, lookback_min))
                    break

            lookback_idx -= 1

    return res


def channel_fit(df: DataFrame) -> Tuple[float, float, float, float]:
    h_slope, h_intercept = linear_regression(
        np.arange(len(df)).reshape(-1, 1), df["high"].values.astype(float)
    )  # type: ignore
    l_slope, l_intercept = linear_regression(
        np.arange(len(df)).reshape(-1, 1), df["low"].values.astype(float)
    )  # type: ignore
    return (h_slope, h_intercept, l_slope, l_intercept)


@st.cache_data()
def get_df_cranges(
    _ts: TradeStation,
    symbols: str,
    firstdate: datetime,
    lastdate: datetime,
    unit: Unit = Unit.DAILY,
    interval: int = 1,
    slope_tol: float = 0.05,
    max_lookback: int = 14,
) -> Iterable[Tuple[DataFrame, List[Tuple[datetime, datetime, float, float]]]]:
    dfs = get_dfs(_ts, symbols.split(","), firstdate, lastdate, unit, interval)
    result = [(df, consolidation_ranges(df, slope_tol, max_lookback)) for df in dfs]
    return result


class BarDirectionLabel(Enum):
    UP = 0
    DOWN = 1
    OUT = 2
    SIDEWAYS = 3


def is_breakout(bar: Series, val: float) -> bool:
    return bar.close > val and bar.open < val


def markov_bardir_state(prev: Series, curr: Series) -> BarDirectionLabel:
    if curr.high > prev.high and curr.low > prev.low:
        return BarDirectionLabel.UP
    elif curr.high < prev.high and curr.low < prev.low:
        return BarDirectionLabel.DOWN
    elif curr.high <= prev.high and curr.low >= prev.low:
        return BarDirectionLabel.SIDEWAYS
    else:
        return BarDirectionLabel.OUT


def markov_label_counts(
    df: DataFrame,
    cranges: List[Tuple[datetime, datetime, float, float]],
    num_next: int = 5,
) -> Dict[int, Counter]:
    """Count state transitions after consolidation range breakouts."""
    result = defaultdict(Counter)

    for _, end_date, high_val, low_val in cranges:
        next_bars = df[df.index > end_date].iloc[0:num_next, :]

        if len(next_bars) < 2:  # Need at least 2 bars for transitions
            continue

        rows = list(next_bars.iterrows())

        # Initialize with range as previous state
        range_series = Series({"high": high_val, "low": low_val})
        prev_state = markov_bardir_state(range_series, rows[0][1]).value

        # Count transitions between consecutive bars
        for transition_idx in range(len(rows) - 1):
            curr_date, curr_series = rows[transition_idx]
            next_date, next_series = rows[transition_idx + 1]

            curr_state = markov_bardir_state(curr_series, next_series).value

            # Count transition from prev_state to curr_state at transition_idx
            result[transition_idx][(curr_state, prev_state)] += 1
            prev_state = curr_state

    return result


def markov_transition_matricies(
    df_cranges: Iterable[
        Tuple[DataFrame, List[Tuple[datetime, datetime, float, float]]]
    ],
    num_next: int = 5,
    alpha: float = 1.0,
) -> Dict[int, np.ndarray]:
    """Build transition matrices with Laplace smoothing."""
    totals = defaultdict(Counter)

    # Aggregate counts across all symbols
    for df, cranges in df_cranges:
        mlcounts = markov_label_counts(df, cranges, num_next)
        for idx, counter in mlcounts.items():
            totals[idx] += counter

    transition_matrices = {}

    for idx, counter in totals.items():
        # Initialize 4x4 matrix (for 4 states: SIDEWAYS, UP, DOWN, OUT)
        matrix = np.full((4, 4), alpha, dtype=float)  # Start with Laplace smoothing

        # Add observed counts
        for (to_state, from_state), count in counter.items():
            if 0 <= from_state < 4 and 0 <= to_state < 4:
                matrix[from_state, to_state] += count

        # Normalize rows to get probabilities
        row_sums = matrix.sum(axis=1, keepdims=True)
        transition_matrices[idx] = matrix / row_sums

    return transition_matrices


def monte_carlo_simulation(
    transition_matrix: np.ndarray,
    start_state: BarDirectionLabel,
    num_steps: int,
    num_simulations: int,
) -> List[Dict[BarDirectionLabel, float]]:
    """
    Run Monte Carlo simulation and return probability distributions over time.

    Returns:
        List of dictionaries where each dict contains the probability of each state
        at that time step. Index 0 is the initial state probabilities.
    """
    # Initialize count matrix: [time_step, state] -> count
    state_counts = np.zeros((num_steps + 1, len(BarDirectionLabel)), dtype=int)

    # Count initial state
    state_counts[0, start_state.value] = num_simulations

    for _ in range(num_simulations):
        current_state = start_state.value

        for step in range(1, num_steps + 1):
            next_state_value = np.random.choice(
                [state.value for state in BarDirectionLabel],
                p=transition_matrix[current_state],
            )
            state_counts[step, next_state_value] += 1
            current_state = next_state_value

    # Convert counts to probabilities
    probabilities_over_time = []
    for step in range(num_steps + 1):
        step_probs = {}
        total_count = state_counts[step].sum()

        for state in BarDirectionLabel:
            step_probs[state] = state_counts[step, state.value] / total_count

        probabilities_over_time.append(step_probs)

    return probabilities_over_time
