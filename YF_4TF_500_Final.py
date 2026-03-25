"""
chart_generator_4tf.py
======================
Reads scrip symbols from 'Scrips_500.xlsx', downloads OHLC data across four
timeframes, and saves a single TradingView-style (Dark theme) PNG per stock.

┌─────────────────────┬─────────────────────┐
│  TOP-LEFT           │  TOP-RIGHT          │
│  Daily  – 100 bars  │  Weekly – 100 bars  │
│  Price + EMA9       │  Price + EMA9       │
│  MACD (12,26,9)     │  MACD (12,26,9)     │
├─────────────────────┼─────────────────────┤
│  BOTTOM-LEFT        │  BOTTOM-RIGHT       │
│  Monthly– 100 bars  │  Hourly – 100 bars  │
│  Price + EMA9       │  Price + EMA9       │
│  MACD (12,26,9)     │  MACD (12,26,9)     │
└─────────────────────┴─────────────────────┘

Requirements:
    pip install yfinance pandas openpyxl matplotlib

Usage:
    Place Scrips_500.xlsx in the same directory and run:
        python chart_generator_4tf.py
"""

import os
import time
import warnings
import traceback
from datetime import datetime, timedelta
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import yfinance as yf

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
EXCEL_FILE   = "Scrips_500.xlsx"
SYMBOL_COL   = None
EXCHANGE_SFX = ".NS"
OUTPUT_DIR   = "YF_4TF_500"

N_BARS       = 100          # bars to plot in each quarter

EMA_PERIOD   = 9
MACD_FAST    = 12
MACD_SLOW    = 26
MACD_SIGNAL  = 9

MAX_RETRIES  = 3
RETRY_DELAY  = 5

# Lookback multipliers to ensure N_BARS are available after fetching
# Daily  : 100 bars × 1.6 calendar days/bar  = ~160 days
# Weekly : 100 bars × 7 days/bar             = ~700 days
# Monthly: fetch max available (~10 yrs) — recently listed stocks may have fewer
# Hourly : 100 bars ÷ 6 bars/day × 1.8 buffer = ~30 days (120 for safety)
LOOKBACK = {
    "1d":  180,
    "1wk": 730,
    "1mo": 3650,   # 10 years – yfinance monthly goes back further but 10yr covers most NSE stocks
    "1h":  120,
}

# ── Dark TradingView palette ─────────────────
STYLE = {
    "bg":          "#131722",
    "panel_bg":    "#1E222D",
    "grid":        "#2A2E39",
    "text":        "#D1D4DC",
    "subtext":     "#787B86",
    "border":      "#2A2E39",
    "zero_line":   "#787B86",
    # ── per-timeframe candle colours ─────────
    # Daily   – classic teal / red
    "d_up":        "#26A69A",
    "d_dn":        "#EF5350",
    "d_ema":       "#FF9800",
    "d_macd":      "#2962FF",
    "d_sig":       "#FF6D00",
    "d_hu":        "#26A69A",
    "d_hd":        "#EF5350",
    # Weekly  – lime / crimson
    "w_up":        "#66BB6A",
    "w_dn":        "#EF5350",
    "w_ema":       "#FFEB3B",
    "w_macd":      "#29B6F6",
    "w_sig":       "#FF7043",
    "w_hu":        "#66BB6A",
    "w_hd":        "#EF5350",
    # Monthly – violet / orange
    "m_up":        "#AB47BC",
    "m_dn":        "#FF7043",
    "m_ema":       "#00BCD4",
    "m_macd":      "#7E57C2",
    "m_sig":       "#FF8A65",
    "m_hu":        "#AB47BC",
    "m_hd":        "#FF7043",
    # Hourly  – yellow / magenta
    "h_up":        "#FFD700",
    "h_dn":        "#FF00FF",
    "h_ema":       "#00FFCC",
    "h_macd":      "#40C4FF",
    "h_sig":       "#FF9E40",
    "h_hu":        "#FFD700",
    "h_hd":        "#FF00FF",
}


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def read_symbols(path, col):
    df = pd.read_excel(path, sheet_name=0)
    if col and col in df.columns:
        return df[col].dropna().astype(str).str.strip().tolist()
    for c in df.columns:
        sample = df[c].dropna().astype(str).str.strip()
        mask   = sample.str.match(r'^[A-Z0-9&\-]{2,20}$')
        if mask.sum() > len(sample) * 0.5:
            print(f"  Auto-detected symbol column: '{c}'")
            return sample[mask].tolist()
    return df.iloc[:, 0].dropna().astype(str).str.strip().tolist()


def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def macd_calc(close, fast=12, slow=26, signal=9):
    ml = ema(close, fast) - ema(close, slow)
    sl = ema(ml, signal)
    return ml, sl, ml - sl


def download_with_retry(ticker, start, end, interval):
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = yf.download(ticker, start=start, end=end,
                             interval=interval, auto_adjust=True,
                             progress=False)
            return df
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                wait = RETRY_DELAY * attempt
                print(f"  ! retry {attempt}/{MAX_RETRIES} in {wait}s …",
                      end="", flush=True)
                time.sleep(wait)
    raise last_exc


def fetch_ohlc(ticker, interval):
    """
    Download and clean OHLC data for a given interval.
    Returns a DataFrame trimmed to the last N_BARS rows (or fewer if
    the stock doesn't have that much history), or None on failure.
    """
    end_dt   = datetime.today() + timedelta(days=1)
    start_dt = end_dt - timedelta(days=LOOKBACK[interval])

    try:
        df = download_with_retry(ticker,
                                 start=start_dt.strftime("%Y-%m-%d"),
                                 end=end_dt.strftime("%Y-%m-%d"),
                                 interval=interval)
        if df is None or df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        df.index = pd.to_datetime(df.index)

        # Hourly: convert UTC → IST, filter market hours, strip tz
        if interval == "1h":
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            df.index = df.index.tz_convert("Asia/Kolkata")
            df = df.between_time("09:15", "15:30")
            df.index = df.index.tz_localize(None)

        # Need at least enough bars for MACD to be meaningful
        min_bars = MACD_SLOW + MACD_SIGNAL + 2
        if len(df) < min_bars:
            return None

        # Use up to N_BARS — if fewer available (e.g. recently listed stock),
        # use whatever is available rather than returning None
        return df.tail(N_BARS)

    except Exception:
        return None


# ─────────────────────────────────────────────
# DRAW ONE QUARTER
# ─────────────────────────────────────────────

def draw_quarter(ax_price, ax_macd, df, label, colors, date_fmt, y_side="right"):
    """
    Draw candlestick + EMA9 on ax_price and MACD on ax_macd.
    y_side : "left" for left column, "right" for right column
    """
    s   = STYLE
    n   = len(df)
    xs  = np.arange(n)

    ema9              = ema(df["Close"], EMA_PERIOD)
    macd_l, sig, hist = macd_calc(df["Close"], MACD_FAST, MACD_SLOW, MACD_SIGNAL)

    # ── style both axes ───────────────────────────────────────────────────────
    for ax in (ax_price, ax_macd):
        ax.set_facecolor(s["panel_bg"])
        ax.tick_params(colors=s["text"], labelsize=7, width=1.0, length=3)
        for spine in ax.spines.values():
            spine.set_edgecolor(s["border"])
        ax.grid(True, color=s["grid"], linewidth=0.5, alpha=0.7)

    # ── minimum body height so doji candles are visible ───────────────────────
    avg_range  = (df["High"] - df["Low"]).mean()
    min_body_h = avg_range * 0.04

    # ── candles ───────────────────────────────────────────────────────────────
    for i, (_, row) in enumerate(df.iterrows()):
        o, h, l, c = row["Open"], row["High"], row["Low"], row["Close"]
        is_bull  = c >= o
        body_col = colors["up"] if is_bull else colors["dn"]
        body_h   = max(abs(c - o), min_body_h)

        ax_price.bar(i, h - l,   bottom=l,        width=0.06,
                     color="#787B86", zorder=2)
        ax_price.bar(i, body_h,  bottom=min(o, c), width=0.65,
                     color=body_col, zorder=3)

    # ── EMA ───────────────────────────────────────────────────────────────────
    ax_price.plot(xs, ema9.values, color=colors["ema"],
                  linewidth=1.8, zorder=4, label=f"EMA {EMA_PERIOD}")

    # ── axes limits ───────────────────────────────────────────────────────────
    ax_price.set_xlim(-1, n + 1)
    pad = (df["High"].max() - df["Low"].min()) * 0.05
    ax_price.set_ylim(df["Low"].min() - pad, df["High"].max() + pad)

    # Y-axis on the correct side per column — eliminates wasted space
    ax_price.yaxis.set_label_position(y_side)
    ax_price.yaxis.tick_right() if y_side == "right" else ax_price.yaxis.tick_left()
    ax_price.set_ylabel("Price (₹)", color=s["text"], fontsize=7)
    ax_price.yaxis.set_tick_params(labelsize=7, labelcolor=s["text"])

    ax_macd.yaxis.set_label_position(y_side)
    ax_macd.yaxis.tick_right() if y_side == "right" else ax_macd.yaxis.tick_left()
    ax_macd.set_ylabel("MACD", color=s["text"], fontsize=7)
    ax_macd.yaxis.set_tick_params(labelsize=6, labelcolor=s["text"])

    # ── last-price pills — always on the RIGHT edge of each axes ─────────────
    # Left column:  pills point right into the wspace gap (wspace is wide enough)
    # Right column: pills point right into the right figure margin
    last_close = df["Close"].iloc[-1]
    last_ema   = ema9.iloc[-1]
    close_col  = colors["up"] if df["Close"].iloc[-1] >= df["Open"].iloc[-1] \
                 else colors["dn"]

    for val, col in [(last_close, close_col), (last_ema, colors["ema"])]:
        ax_price.annotate(f"₹{val:,.2f}",
                          xy=(1, val), xycoords=("axes fraction", "data"),
                          xytext=(4, 0), textcoords="offset points",
                          fontsize=7.5, fontweight="bold", color="#131722",
                          ha="left", va="center",
                          bbox=dict(boxstyle="round,pad=0.3",
                                    facecolor=col, edgecolor="none", alpha=0.97),
                          annotation_clip=False)

    # ── legend ────────────────────────────────────────────────────────────────
    leg = [mpatches.Patch(facecolor=colors["up"], label="Bullish"),
           mpatches.Patch(facecolor=colors["dn"], label="Bearish"),
           Line2D([0],[0], color=colors["ema"], linewidth=1.5,
                  label=f"EMA {EMA_PERIOD}")]
    ax_price.legend(handles=leg, loc="upper left", fontsize=7,
                    framealpha=0.7, facecolor=s["bg"],
                    edgecolor=s["border"], labelcolor=s["text"])

    # ── TIMEFRAME LABEL – axes title, always visible above the quadrant ───────
    ax_price.set_title(f"  {label}  ",
                       loc="left",
                       color=s["bg"],
                       fontsize=10, fontweight="bold",
                       pad=3,
                       bbox=dict(boxstyle="round,pad=0.35",
                                 facecolor=colors["ema"],
                                 edgecolor="none",
                                 alpha=0.95))

    # ── MACD ──────────────────────────────────────────────────────────────────
    hcols = [colors["hu"] if v >= 0 else colors["hd"] for v in hist.values]
    ax_macd.bar(xs, hist.values, color=hcols, alpha=0.80, width=0.65, zorder=2)
    ax_macd.plot(xs, macd_l.values, color=colors["macd"], linewidth=1.0,
                 zorder=3, label="MACD")
    ax_macd.plot(xs, sig.values,    color=colors["sig"],  linewidth=0.9,
                 zorder=3, label="Signal")
    ax_macd.axhline(0, color=s["zero_line"], linewidth=0.6, linestyle="--")
    ax_macd.legend(loc="upper left", fontsize=6.5, framealpha=0.6,
                   facecolor=s["bg"], edgecolor=s["border"],
                   labelcolor=s["text"])

    # ── x-axis labels ─────────────────────────────────────────────────────────
    step = max(n // 8, 1)
    ax_macd.set_xticks(xs[::step])
    ax_macd.set_xticklabels(
        [df.index[i].strftime(date_fmt) for i in range(0, n, step)],
        rotation=25, ha="right", fontsize=7, color=s["text"])
    plt.setp(ax_price.get_xticklabels(), visible=False)

    # ── % change caption – bottom-right of price panel ────────────────────────
    lc   = df["Close"].iloc[-1]
    fc   = df["Close"].iloc[0]
    pct  = (lc - fc) / fc * 100
    sign = "+" if pct >= 0 else ""
    ccol = colors["up"] if pct >= 0 else colors["dn"]
    ax_price.text(0.99, 0.02, f"{sign}{pct:.2f}%  ({n} bars)",
                  transform=ax_price.transAxes,
                  color=ccol, fontsize=7.5, fontweight="bold",
                  ha="right", va="bottom")


# ─────────────────────────────────────────────
# MASTER CHART  (4 quarters)
# ─────────────────────────────────────────────

def plot_chart(symbol, d_df, w_df, m_df, h_df, output_path):
    """
    d_df, w_df, m_df, h_df : DataFrames for daily/weekly/monthly/hourly
                              Any may be None → that quarter shows "No data"
    """
    s = STYLE

    # ── Figure: 2 rows × 2 cols, each cell split into price(70%) + MACD(30%) ─
    # We use GridSpec with 4 rows and 2 cols:
    #   rows 0,1 → top half  (row 0 = price, row 1 = MACD)
    #   rows 2,3 → bot half  (row 2 = price, row 3 = MACD)
    #   cols 0   → left
    #   cols 1   → right
    # ── Figure sized for 1920×1080 Full HD screen ────────────────────────────
    # figsize(19.2, 10.4) × dpi(100) = 1920 × 1040 px
    # 1040 instead of 1080 leaves ~40px for Windows taskbar so chart fills
    # the screen without scrolling when viewed at 100% zoom
    fig = plt.figure(figsize=(19.2, 10.4), dpi=100, facecolor=s["bg"])
    fig.patch.set_facecolor(s["bg"])

    gs = gridspec.GridSpec(
        4, 2,
        height_ratios=[7, 3, 7, 3],
        width_ratios=[1, 1],
        hspace=0.10,
        wspace=0.14,       # wide enough for left-col right-side pill labels
        top=0.93, bottom=0.05,
        left=0.02, right=0.97,
    )

    # Quarter axes
    ax_d_p  = fig.add_subplot(gs[0, 0])   # Daily   – price
    ax_d_m  = fig.add_subplot(gs[1, 0], sharex=ax_d_p)   # Daily   – MACD
    ax_w_p  = fig.add_subplot(gs[0, 1])   # Weekly  – price
    ax_w_m  = fig.add_subplot(gs[1, 1], sharex=ax_w_p)   # Weekly  – MACD
    ax_mo_p = fig.add_subplot(gs[2, 0])   # Monthly – price
    ax_mo_m = fig.add_subplot(gs[3, 0], sharex=ax_mo_p)  # Monthly – MACD
    ax_h_p  = fig.add_subplot(gs[2, 1])   # Hourly  – price
    ax_h_m  = fig.add_subplot(gs[3, 1], sharex=ax_h_p)   # Hourly  – MACD

    # ── colour maps per timeframe ─────────────────────────────────────────────
    tf_colors = {
        "d":  dict(up=s["d_up"],  dn=s["d_dn"],  ema=s["d_ema"],
                   macd=s["d_macd"], sig=s["d_sig"], hu=s["d_hu"], hd=s["d_hd"]),
        "w":  dict(up=s["w_up"],  dn=s["w_dn"],  ema=s["w_ema"],
                   macd=s["w_macd"], sig=s["w_sig"], hu=s["w_hu"], hd=s["w_hd"]),
        "mo": dict(up=s["m_up"],  dn=s["m_dn"],  ema=s["m_ema"],
                   macd=s["m_macd"], sig=s["m_sig"], hu=s["m_hu"], hd=s["m_hd"]),
        "h":  dict(up=s["h_up"],  dn=s["h_dn"],  ema=s["h_ema"],
                   macd=s["h_macd"], sig=s["h_sig"], hu=s["h_hu"], hd=s["h_hd"]),
    }

    # ── helper: show "No data" placeholder ───────────────────────────────────
    def no_data(ax_p, ax_m, label):
        for ax in (ax_p, ax_m):
            ax.set_facecolor(s["panel_bg"])
            for spine in ax.spines.values():
                spine.set_edgecolor(s["border"])
        ax_p.text(0.5, 0.5, f"{label}\nNo data",
                  transform=ax_p.transAxes,
                  color=s["subtext"], fontsize=12,
                  ha="center", va="center")

    # ── draw each quarter ─────────────────────────────────────────────────────
    quarters = [
        (ax_d_p,  ax_d_m,  d_df,  "Daily",   "d",  "%d %b '%y",   "left"),
        (ax_w_p,  ax_w_m,  w_df,  "Weekly",  "w",  "%d %b '%y",   "right"),
        (ax_mo_p, ax_mo_m, m_df,  "Monthly", "mo", "%b '%y",      "left"),
        (ax_h_p,  ax_h_m,  h_df,  "Hourly",  "h",  "%d %b %H:%M", "right"),
    ]

    for ax_p, ax_m, df, label, key, dfmt, yside in quarters:
        if df is not None and len(df) >= MACD_SLOW + MACD_SIGNAL + 2:
            draw_quarter(ax_p, ax_m, df, label, tf_colors[key], dfmt, yside)
        else:
            no_data(ax_p, ax_m, label)

    # ── master title ──────────────────────────────────────────────────────────
    latest = (d_df.index[-1].strftime("%d %b %Y")
              if d_df is not None and not d_df.empty else "–")
    lc = (d_df["Close"].iloc[-1] if d_df is not None and not d_df.empty else 0)

    fig.text(0.03, 0.965,
             f"{symbol}  |  NSE  |  Multi-Timeframe Chart",
             color=s["text"], fontsize=12, fontweight="bold")
    fig.text(0.03, 0.945,
             f"₹{lc:,.2f}  |  {N_BARS} bars per timeframe",
             color=s["subtext"], fontsize=8)
    fig.text(0.97, 0.965,
             f"Latest: {latest}",
             color=s["text"], fontsize=8, ha="right", fontweight="bold")
    fig.text(0.97, 0.945,
             f"EMA {EMA_PERIOD}  |  MACD ({MACD_FAST},{MACD_SLOW},{MACD_SIGNAL})"
             f"  |  Data: yfinance",
             color=s["subtext"], fontsize=7, ha="right")

    # ── quarter border lines ──────────────────────────────────────────────────
    # Horizontal divider between top and bottom halves
    fig.add_artist(plt.Line2D([0.02, 0.97], [0.50, 0.50],
                              transform=fig.transFigure,
                              color=s["border"], linewidth=1.5, alpha=0.8))
    # Vertical divider between left and right columns
    fig.add_artist(plt.Line2D([0.50, 0.50], [0.05, 0.93],
                              transform=fig.transFigure,
                              color=s["border"], linewidth=1.5, alpha=0.8))

    plt.savefig(output_path, dpi=100,
                facecolor=s["bg"], edgecolor="none")
    plt.close(fig)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n{'='*62}")
    print(f"  Multi-TF Chart Generator  –  NSE  |  TradingView Dark")
    print(f"  Quarters: Daily · Weekly · Monthly · Hourly  |  {N_BARS} bars each")
    print(f"{'='*62}")

    if not os.path.exists(EXCEL_FILE):
        print(f"\n[ERROR] '{EXCEL_FILE}' not found.")
        return

    symbols = read_symbols(EXCEL_FILE, SYMBOL_COL)
    print(f"\n  Loaded {len(symbols)} symbols from '{EXCEL_FILE}'")

    success, failed = [], []

    for i, sym in enumerate(symbols, 1):
        ticker = sym if sym.endswith(EXCHANGE_SFX) else sym + EXCHANGE_SFX
        print(f"\n[{i:>3}/{len(symbols)}]  {ticker:<22}", end="", flush=True)

        try:
            d_df  = fetch_ohlc(ticker, "1d")
            w_df  = fetch_ohlc(ticker, "1wk")
            m_df  = fetch_ohlc(ticker, "1mo")
            h_df  = fetch_ohlc(ticker, "1h")

            counts = (f"D:{len(d_df) if d_df is not None else 0}"
                      f"  W:{len(w_df) if w_df is not None else 0}"
                      f"  M:{len(m_df) if m_df is not None else 0}"
                      f"  H:{len(h_df) if h_df is not None else 0}")
            print(f"  {counts}", end="", flush=True)

            out = os.path.join(OUTPUT_DIR, f"{sym}.png")
            plot_chart(sym, d_df, w_df, m_df, h_df, out)
            print(f"  →  {out}")
            success.append(sym)

        except Exception:
            print(f"  ✗  Error")
            traceback.print_exc()
            failed.append(sym)

    print(f"\n{'='*62}")
    print(f"  Done!  {len(success)} charts saved to '{OUTPUT_DIR}/'")
    if failed:
        print(f"  Failed ({len(failed)}): {', '.join(failed[:20])}"
              + (" …" if len(failed) > 20 else ""))
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()