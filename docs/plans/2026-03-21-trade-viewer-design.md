# Trade Viewer Design

## Problem

After running QC backtests, there's no easy way to visually inspect individual trades at different chart resolutions. The existing Plotly notebook viewer only shows daily bars and doesn't support resolution switching.

## Goal

Add a trades page to the tradestation-streamlit app that loads QC backtest trade CSVs from `~/Downloads`, lets you select individual trades, and view them on candlestick charts at daily, hourly, or minute resolution with auto-bounded date ranges.

## UI Flow

```
Sidebar:
  [Dropdown] Backtest file: scans ~/Downloads for *_trades.csv
  [Dropdown] Trade: {rank}. {ticker} {entry_date} (${pnl})
  [Radio] Resolution: Daily | Hourly | Minute

Main area:
  Candlestick chart with entry/exit vertical markers
  Title: {ticker} — {WIN/LOSS} (P&L: ${pnl}) | Entry {date} → Exit {date}
```

- Auto-scans `~/Downloads` for `*_trades.csv` on page load
- Selecting a file parses the QC trades CSV, populates trade dropdown
- Selecting a trade auto-sets date bounds and fetches bars from TradeStation
- Resolution buttons re-fetch without changing selected trade

## Trade CSV Parsing

Group legs into spreads by `(entry_dt, underlying)`. Each trade has: underlying, entry_time, exit_time, total_pnl, is_win, legs, direction.

## Predefined Date Bounds

- **Daily**: entry - 5 trading days → exit + 2 trading days
- **Hourly**: entry - 1 trading day → exit + 1 trading day
- **Minute**: entry day open → exit day close

## Chart Rendering

Reuse existing `utils/plot.py` candlestick rendering. Add:
- Blue vertical dashed line at entry_time with "Entry" label
- Red vertical dashed line at exit_time with "Exit" label
- Existing features (volume bars, dark theme, scroll zoom) carry over

TradeStation bar fetch:
- Daily: unit=DAILY, interval=1
- Hourly: unit=MINUTE, interval=60
- Minute: unit=MINUTE, interval=1

## File Changes

- Create: `pages/trades.py` — trade viewer page
- Modify: `utils/data.py` — add `parse_trades_csv()` for local file reading
- Modify: `utils/plot.py` — add `trade_markers()` for entry/exit overlays
