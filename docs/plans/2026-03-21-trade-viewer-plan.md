# Trade Viewer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a trades page to tradestation-streamlit that auto-loads QC backtest trade CSVs from `~/Downloads`, lets you select individual trades, and view them on candlestick charts at daily/hourly/minute resolution with auto-bounded date ranges and entry/exit markers.

**Architecture:** New `pages/trades.py` Streamlit page. Sidebar: file selector (auto-scans ~/Downloads), trade dropdown, resolution radio. Main area: candlestick chart from TradeStation API with entry/exit vertical markers. Trade parsing added to `utils/data.py`, marker rendering added to `utils/plot.py`.

**Tech Stack:** Streamlit, Plotly, TradeStation API (tradestation-python), pandas

---

### Task 1: Add `parse_trades_csv` to `utils/data.py`

**Files:**
- Modify: `utils/data.py`

**Step 1: Add the function**

Add at the end of `utils/data.py`. Reads a local QC backtest trades CSV file and groups legs into spreads.

```python
from pathlib import Path


def scan_trade_files(directory: str = "~/Downloads") -> list[Path]:
    """Scan directory for QC backtest trade CSV files."""
    path = Path(directory).expanduser()
    return sorted(path.glob("*_trades.csv"), key=lambda p: p.stat().st_mtime, reverse=True)


def parse_trades_csv(filepath: Path) -> DataFrame:
    """Parse a QC backtest trades CSV into a DataFrame of spreads.

    Groups individual legs by (entry_time, underlying) and aggregates
    P&L, fees, win status, and leg details.
    """
    import pandas as pd

    bt_raw = pd.read_csv(filepath)
    bt_raw["underlying"] = bt_raw["Symbols"].str.strip().str.split().str[0]
    bt_raw["entry_dt"] = pd.to_datetime(bt_raw["Entry Time"])
    bt_raw["exit_dt"] = pd.to_datetime(bt_raw["Exit Time"])

    return (
        bt_raw.groupby(["entry_dt", "underlying"])
        .agg(
            exit_time=("exit_dt", "max"),
            total_pnl=("P&L", "sum"),
            total_fees=("Fees", "sum"),
            is_win=("IsWin", "max"),
            legs=("Symbols", list),
            direction=("Direction", list),
        )
        .reset_index()
        .rename(columns={"entry_dt": "entry_time"})
        .sort_values("entry_time")
        .reset_index(drop=True)
    )
```

**Step 2: Commit**

```bash
git add utils/data.py
git commit -m "feat(data): add scan_trade_files and parse_trades_csv"
```

---

### Task 2: Add `add_trade_markers_to_fig` to `utils/plot.py`

**Files:**
- Modify: `utils/plot.py`

**Step 1: Add the function**

Add at the end of `utils/plot.py`. Draws vertical entry/exit lines on the chart.

```python
def add_trade_markers_to_fig(
    fig: Figure,
    entry_time: datetime,
    exit_time: datetime,
    entry_color: str = "rgba(0, 150, 255, 0.8)",
    exit_color: str = "rgba(255, 80, 80, 0.8)",
) -> Figure:
    """Add vertical entry/exit markers to a candlestick chart."""
    for dt, color, label in [
        (entry_time, entry_color, "Entry"),
        (exit_time, exit_color, "Exit"),
    ]:
        fig.add_vline(
            x=dt,
            line_dash="dot",
            line_color=color,
            annotation_text=label,
            annotation_position="top",
            annotation_font_color=color,
            secondary_y=True,
        )
    return fig
```

**Step 2: Commit**

```bash
git add utils/plot.py
git commit -m "feat(plot): add trade entry/exit marker overlays"
```

---

### Task 3: Create `pages/trades.py`

**Files:**
- Create: `pages/trades.py`

**Step 1: Write the page**

```python
from datetime import timedelta
from pathlib import Path

import streamlit as st
from pandas import DataFrame, Timestamp
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
    entry = Timestamp(row["entry_time"]).to_pydatetime()
    exit_ = Timestamp(row["exit_time"]).to_pydatetime()
    before, after = BOUNDS[resolution]

    if resolution == "Minute":
        # Entry day open to exit day close
        firstdate = entry.replace(hour=9, minute=30, second=0)
        lastdate = exit_.replace(hour=16, minute=0, second=0)
    else:
        firstdate = entry - before
        lastdate = exit_ + after

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
        selected_idx = st.selectbox("Backtest", range(len(file_labels)), format_func=lambda i: file_labels[i])
        trades_df = parse_trades_csv(trade_files[selected_idx])

        if trades_df.empty:
            st.warning("No trades found in file")
            st.stop()

        # Trade selection
        labels = [trade_label(i, row) for i, row in trades_df.iterrows()]
        trade_idx = st.selectbox("Trade", range(len(labels)), format_func=lambda i: labels[i])
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

    entry_time = Timestamp(trade["entry_time"]).to_pydatetime()
    exit_time = Timestamp(trade["exit_time"]).to_pydatetime()
    fig = add_trade_markers_to_fig(fig, entry_time, exit_time)

    win = "WIN" if trade["is_win"] else "LOSS"
    fig.update_layout(
        title=f"{symbol} — {win} (P&L: ${trade['total_pnl']:.0f}) | Entry {entry_time.date()} → Exit {exit_time.date()}",
    )

    pconfig = configure_plotly()
    st.plotly_chart(fig, use_container_width=True, config=pconfig)

    # Trade details
    with st.expander("Trade Details"):
        st.write(f"**Entry:** {entry_time}")
        st.write(f"**Exit:** {exit_time}")
        st.write(f"**P&L:** ${trade['total_pnl']:.2f}")
        st.write(f"**Fees:** ${trade['total_fees']:.2f}")
        st.write(f"**Legs:** {trade['legs']}")
```

**Step 2: Verify the app runs**

```bash
cd ~/projects/tradestation-streamlit
~/miniconda3/envs/tradestation/bin/streamlit run main.py
```

Navigate to the Trades page in the browser. Verify:
- Dropdown shows `*_trades.csv` files from `~/Downloads`
- Selecting a file populates the trade dropdown
- Selecting a trade + resolution shows the chart with entry/exit markers
- Switching resolution re-fetches and re-renders

**Step 3: Commit**

```bash
git add pages/trades.py
git commit -m "feat(pages): add trades page with multi-resolution trade viewer"
```

---

### Task 4: Lint and verify

**Step 1: Run lint**

```bash
~/miniconda3/envs/tradestation/bin/ruff check --fix . && ~/miniconda3/envs/tradestation/bin/ruff format .
```

**Step 2: Run the app end-to-end**

```bash
~/miniconda3/envs/tradestation/bin/streamlit run main.py
```

Test:
- Select a backtest file with known trades
- Verify entry/exit markers align with trade times
- Switch between Daily → Hourly → Minute
- Verify date bounds auto-adjust per resolution

**Step 3: Commit if any lint changes**

```bash
git add -A
git commit -m "chore: lint and format"
```
