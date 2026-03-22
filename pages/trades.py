from datetime import timedelta

import streamlit as st
from pandas import Timestamp
from tradestation_python import TradeStation
from tradestation_python.types.enums import Unit

from utils.data import (
    convert_bars_to_df,
    convert_df_to_fig,
    get_bars,
    parse_trades_csv,
    scan_trade_files,
)
from utils.plot import add_trade_markers_to_fig, configure_plotly

RESOLUTIONS = {
    "Daily": (Unit.DAILY, 1),
    "Hourly": (Unit.MINUTE, 60),
    "Minute": (Unit.MINUTE, 1),
}

# Predefined date bounds per resolution: (before_entry, after_exit)
BOUNDS = {
    "Daily": (timedelta(days=7), timedelta(days=3)),
    "Hourly": (timedelta(days=1), timedelta(days=1)),
    "Minute": (timedelta(hours=0), timedelta(hours=0)),
}


def trade_label(i: int, row) -> str:
    win = "WIN" if row["is_win"] else "LOSS"
    entry = Timestamp(row["entry_time"]).strftime("%Y-%m-%d")
    pnl = row["total_pnl"]
    return f"{i + 1}. {row['underlying']} {entry} ${pnl:.0f} ({win})"


def compute_bounds(row, resolution: str) -> tuple:
    """Compute date bounds in UTC for the TradeStation API."""
    import pytz

    eastern = pytz.timezone("US/Eastern")
    entry_et = Timestamp(row["entry_time"]).tz_convert("US/Eastern")
    exit_et = Timestamp(row["exit_time"]).tz_convert("US/Eastern")
    before, after = BOUNDS[resolution]

    if resolution == "Minute":
        # Build ET market hours, convert back to UTC for the API
        firstdate = (
            entry_et.replace(hour=9, minute=30, second=0)
            .tz_convert("UTC")
            .tz_localize(None)
            .to_pydatetime()
        )
        lastdate = (
            exit_et.replace(hour=16, minute=0, second=0)
            .tz_convert("UTC")
            .tz_localize(None)
            .to_pydatetime()
        )
    else:
        firstdate = entry_et.tz_localize(None).to_pydatetime() - before
        lastdate = exit_et.tz_localize(None).to_pydatetime() + after

    return firstdate, lastdate


if __name__ == "__main__":
    ts = TradeStation()

    with st.sidebar:
        # File selection
        trade_files = scan_trade_files()
        if not trade_files:
            st.warning("No *_trades.csv files found in ~/Downloads")
            st.stop()

        file_labels = [f.stem for f in trade_files]
        selected_idx = st.selectbox(
            "Backtest", range(len(file_labels)), format_func=lambda i: file_labels[i]
        )
        trades_df = parse_trades_csv(trade_files[selected_idx])

        if trades_df.empty:
            st.warning("No trades found in file")
            st.stop()

        # Trade selection (sorted by P&L, worst first)
        trades_df = trades_df.sort_values("total_pnl").reset_index(drop=True)
        labels = [trade_label(i, row) for i, row in trades_df.iterrows()]
        trade_idx = st.selectbox(
            "Trade", range(len(labels)), format_func=lambda i: labels[i]
        )
        trade = trades_df.iloc[trade_idx]

        # Resolution
        resolution = st.radio("Resolution", list(RESOLUTIONS.keys()), horizontal=True)

    # Fetch bars from TradeStation
    unit, interval = RESOLUTIONS[resolution]
    firstdate, lastdate = compute_bounds(trade, resolution)
    symbol = trade["underlying"]

    try:
        bars = get_bars(ts, symbol, firstdate, lastdate, unit, interval)
    except Exception as e:
        st.error(f"Error fetching bars: {e}")
        get_bars.clear()
        st.stop()

    if not bars:
        st.warning(f"No bars returned for {symbol}")
        st.stop()

    df = convert_bars_to_df(bars)

    # Build chart
    fconfig = {"volume": "rgba(132, 170, 183, 0.4)"}
    fig = convert_df_to_fig(df, fconfig)

    entry_time = (
        Timestamp(trade["entry_time"])
        .tz_convert("US/Eastern")
        .tz_localize(None)
        .to_pydatetime()
    )
    exit_time = (
        Timestamp(trade["exit_time"])
        .tz_convert("US/Eastern")
        .tz_localize(None)
        .to_pydatetime()
    )
    fig = add_trade_markers_to_fig(fig, entry_time, exit_time)

    win = "WIN" if trade["is_win"] else "LOSS"
    fig.update_layout(
        title=f"{symbol} — {win} (P&L: ${trade['total_pnl']:.0f}) | Entry {entry_time.date()} → Exit {exit_time.date()}",
    )

    pconfig = configure_plotly()
    st.plotly_chart(fig, width="stretch", config=pconfig)

    # Trade details
    with st.expander("Trade Details"):
        st.write(f"**Entry:** {entry_time}")
        st.write(f"**Exit:** {exit_time}")
        st.write(f"**P&L:** ${trade['total_pnl']:.2f}")
        st.write(f"**Fees:** ${trade['total_fees']:.2f}")
        st.write(f"**Legs:** {trade['legs']}")
