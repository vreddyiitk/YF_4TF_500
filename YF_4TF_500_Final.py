"""
chart_generator_4tf_nse.py
==========================
Combines:
  • STEP 1  — Live NSE "Stocks Traded" scrape     (from nse_pipeline.py)
  • STEP 2  — Filter EQ series, Value > ₹10 Cr    (from nse_pipeline.py)
  • STEP 3  — 4-Timeframe charts per stock         (from chart_generator_4tf.py)

Each chart contains 4 quadrants:
  ┌─────────────────────┬─────────────────────┐
  │  Daily  – 100 bars  │  Weekly – 100 bars  │
  │  Price + EMA9       │  Price + EMA9       │
  │  MACD (12,26,9)     │  MACD (12,26,9)     │
  ├─────────────────────┼─────────────────────┤
  │  Monthly– 100 bars  │  Hourly – 100 bars  │
  │  Price + EMA9       │  Price + EMA9       │
  │  MACD (12,26,9)     │  MACD (12,26,9)     │
  └─────────────────────┴─────────────────────┘

  • White background, green/red candles across all timeframes
  • Green ● below bar when MACD histogram turns positive
  • Red   ● above bar when MACD histogram turns negative

Requirements:
    pip install selenium webdriver-manager yfinance pandas openpyxl matplotlib

Usage:
    python chart_generator_4tf_nse.py                  # headless browser
    python chart_generator_4tf_nse.py --visible        # visible browser
    python chart_generator_4tf_nse.py --from-csv FILE  # skip browser
    python chart_generator_4tf_nse.py --min-value 50   # change ₹ Cr threshold
"""

import sys
import os
import glob
import json
import time
import shutil
import datetime
import tempfile
import argparse
import traceback
import warnings
from datetime import timedelta

import pandas as pd
import numpy as np

# ── Selenium ─────────────────────────────────────────────────────────────────
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, WebDriverException
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_OK = True
except ImportError:
    SELENIUM_OK = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import yfinance as yf

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════
#  CONFIG — NSE scrape  (from nse_pipeline.py)
# ═══════════════════════════════════════════════════════════════

TRADED_VALUE_MIN_CR = 10       # ₹ Crore minimum traded value filter

NSE_HOME  = "https://www.nseindia.com"
NSE_PAGE  = "https://www.nseindia.com/market-data/stocks-traded"
NSE_APIS  = [
    "https://www.nseindia.com/api/live-analysis-stocksTraded",
    "https://www.nseindia.com/json/liveAnalysis/stocks-traded.json",
]

PAGE_WAIT     = 30
API_SETTLE    = 10
DOWNLOAD_WAIT = 40

SYNC_XHR = """
var xhr = new XMLHttpRequest();
xhr.open('GET', arguments[0], false);
xhr.setRequestHeader('Accept', 'application/json, text/plain, */*');
xhr.setRequestHeader('X-Requested-With', 'XMLHttpRequest');
xhr.setRequestHeader('Referer',
    'https://www.nseindia.com/market-data/stocks-traded');
try {
    xhr.send(null);
    return {status: xhr.status, body: xhr.responseText};
} catch(e) {
    return {status: -1, body: e.toString()};
}
"""

# ═══════════════════════════════════════════════════════════════
#  CONFIG — Chart  (from chart_generator_4tf.py)
# ═══════════════════════════════════════════════════════════════

EXCHANGE_SFX = ".NS"
OUTPUT_DIR   = "YF_4TF_NSE"

N_BARS      = 100
EMA_PERIOD  = 9
MACD_FAST   = 12
MACD_SLOW   = 26
MACD_SIGNAL = 9
MAX_RETRIES = 3
RETRY_DELAY = 5

LOOKBACK = {
    "1d":  180,
    "1wk": 730,
    "1mo": 3650,
    "1h":  120,
}

STYLE = {
    "bg":             "#FFFFFF",
    "panel_bg":       "#FAFAFA",
    "grid":           "#E0E0E0",
    "text":           "#1A1A2E",
    "subtext":        "#5A5A72",
    "border":         "#CCCCCC",
    "zero_line":      "#999999",
    "macd_turn_bull": "#00C853",
    "macd_turn_bear": "#D50000",
    "up":             "#26A69A",
    "dn":             "#EF5350",
    "d_ema":  "#E65100", "d_macd": "#1565C0", "d_sig": "#E65100",
    "d_hu":   "#26A69A", "d_hd":   "#EF5350",
    "w_ema":  "#6A1B9A", "w_macd": "#0277BD", "w_sig": "#AD1457",
    "w_hu":   "#26A69A", "w_hd":   "#EF5350",
    "m_ema":  "#00695C", "m_macd": "#4527A0", "m_sig": "#BF360C",
    "m_hu":   "#26A69A", "m_hd":   "#EF5350",
    "h_ema":  "#1B5E20", "h_macd": "#0D47A1", "h_sig": "#B71C1C",
    "h_hu":   "#26A69A", "h_hd":   "#EF5350",
}


# ═══════════════════════════════════════════════════════════════
#  STEP 1A — BROWSER SETUP  (from nse_pipeline.py)
# ═══════════════════════════════════════════════════════════════

def build_driver(headless: bool, download_dir: str):
    if not SELENIUM_OK:
        print("[ERROR] selenium / webdriver-manager not installed.")
        print("  Fix: pip install selenium webdriver-manager")
        sys.exit(1)

    opts = Options()
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--disable-extensions")
    opts.add_argument("--disable-setuid-sandbox")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    opts.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
    if headless:
        opts.add_argument("--headless=new")

    opts.add_experimental_option("prefs", {
        "download.default_directory":   download_dir,
        "download.prompt_for_download": False,
        "download.directory_upgrade":   True,
        "safebrowsing.enabled":         True,
    })

    try:
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=opts,
        )
        print("  ✔  ChromeDriver ready (webdriver-manager)")
        return driver
    except Exception as e:
        print(f"  [WARN] webdriver-manager failed: {e}")

    try:
        driver = webdriver.Chrome(options=opts)
        print("  ✔  ChromeDriver ready (system)")
        return driver
    except Exception as e:
        print(f"\n[ERROR] Chrome unavailable: {e}")
        sys.exit(1)


def patch_driver(driver):
    try:
        driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": "Object.defineProperty(navigator,'webdriver',"
                       "{get:()=>undefined});"}
        )
    except Exception:
        pass


def warm_session(driver):
    print("  [1/3]  Loading NSE homepage …")
    driver.get(NSE_HOME)
    time.sleep(4)
    print(f"         Cookies: {[c['name'] for c in driver.get_cookies()]}")

    print("  [2/3]  Loading Stocks Traded page …")
    driver.get(NSE_PAGE)
    try:
        WebDriverWait(driver, PAGE_WAIT).until(
            EC.presence_of_element_located(
                (By.XPATH,
                 "//*[@id='cm_9'] | "
                 "//h2[contains(text(),'Stocks Traded')] | "
                 "//*[contains(text(),'Stocks Traded') and "
                 "    not(contains(@class,'nav'))]")
            )
        )
        print("         Page loaded ✔")
    except TimeoutException:
        print("         [WARN] timeout — continuing")

    print(f"         Settling {API_SETTLE}s for XHR to complete …")
    time.sleep(API_SETTLE)
    print(f"         Cookies: {[c['name'] for c in driver.get_cookies()]}")


# ═══════════════════════════════════════════════════════════════
#  STEP 1B — XHR DATA FETCH  (from nse_pipeline.py)
# ═══════════════════════════════════════════════════════════════

def _find_records_in_json(payload) -> list:
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        if any(k in payload[0] for k in ("symbol", "Symbol", "SYMBOL")):
            return payload
    if isinstance(payload, dict):
        for key in ("data", "stocksTradedData", "result", "rows",
                    "stockData", "DATA", "records", "stocks", "dataList"):
            val = payload.get(key)
            if isinstance(val, list) and val and isinstance(val[0], dict):
                return val
    return []


def fetch_via_xhr(driver) -> list:
    print("  [3/3]  Calling NSE API via sync XHR …")
    for url in NSE_APIS:
        print(f"         → {url}")
        try:
            res    = driver.execute_script(SYNC_XHR, url)
            status = res.get("status", -1)
            body   = res.get("body", "")
            print(f"           HTTP {status}  |  {len(body):,} chars")
            if status != 200 or not body:
                continue
            payload = json.loads(body)
            if isinstance(payload, dict):
                print(f"           JSON keys: {list(payload.keys())}")
            records = _find_records_in_json(payload)
            if records:
                print(f"  ✔  {len(records)} records via XHR")
                print(f"     Fields: {list(records[0].keys())[:10]}")
                return records
            else:
                print("           No stock list found in response")
        except json.JSONDecodeError as e:
            print(f"           JSON error: {e}")
        except Exception as e:
            print(f"           Error: {e}")
    return []


# ═══════════════════════════════════════════════════════════════
#  STEP 1C — CSV BUTTON FALLBACK  (from nse_pipeline.py)
# ═══════════════════════════════════════════════════════════════

def _wait_for_csv(dl_dir: str) -> str:
    print(f"         Waiting {DOWNLOAD_WAIT}s for file", end="", flush=True)
    deadline = time.time() + DOWNLOAD_WAIT
    while time.time() < deadline:
        time.sleep(1)
        print(".", end="", flush=True)
        files = [
            f for f in
            glob.glob(os.path.join(dl_dir, "*.csv")) +
            glob.glob(os.path.join(dl_dir, "*.CSV"))
            if not f.endswith(".crdownload")
        ]
        if files:
            latest = max(files, key=os.path.getmtime)
            print(f"\n  ✔  {os.path.basename(latest)}")
            return latest
    print("\n  Timed out.")
    return ""


def fetch_via_csv_button(driver, dl_dir: str) -> str:
    print("  CSV button fallback …")
    xpaths = [
        "//a[contains(@onclick,'StocksTraded-download')]",
        "//a[contains(@onclick,'StocksTraded')]",
        ".//a[.//img[contains(@src,'xls') or contains(@src,'csv')]]",
        "//a[contains(@onclick,'download') and "
        "    not(contains(@onclick,'First')) and "
        "    not(contains(@onclick,'Prev')) and "
        "    not(contains(@onclick,'Next'))]",
    ]
    for xpath in xpaths:
        try:
            for el in driver.find_elements(By.XPATH, xpath):
                if el.is_displayed():
                    print(f"  Clicking: {el.get_attribute('outerHTML')[:100]}")
                    driver.execute_script(
                        "arguments[0].scrollIntoView(true);", el)
                    time.sleep(0.5)
                    driver.execute_script("arguments[0].click();", el)
                    path = _wait_for_csv(dl_dir)
                    if path:
                        return path
        except Exception:
            continue
    try:
        driver.execute_script("downloadCSV('StocksTraded-download');")
        return _wait_for_csv(dl_dir)
    except Exception as e:
        print(f"  JS call failed: {e}")
    return ""


# ═══════════════════════════════════════════════════════════════
#  STEP 1D — NORMALISE  (from nse_pipeline.py)
# ═══════════════════════════════════════════════════════════════

def safe_num(v) -> float:
    try:
        return float(str(v).replace(",", "").replace("–", "0")
                     .replace("−", "0").strip())
    except Exception:
        return 0.0


def normalise_json(records: list) -> pd.DataFrame:
    rows = []
    for d in records:
        sym = str(d.get("symbol", d.get("Symbol", ""))).strip()
        if not sym:
            continue
        tv_raw = safe_num(d.get("totalTradedValue", d.get("tradedValue", 0)))
        vol    = safe_num(d.get("totalTradedVolume", d.get("tradedQuantity", 0)))
        rows.append({
            "Symbol":           sym,
            "Company":          str(d.get("companyName", "")).strip(),
            "Series":           str(d.get("series", "EQ")).strip(),
            "LTP (₹)":          round(safe_num(d.get("lastPrice",
                                     d.get("closePrice", 0))), 2),
            "% Change":         round(safe_num(d.get("pChange", 0)), 2),
            "Mkt Cap (₹ Cr)":   round(safe_num(d.get("marketCap",
                                     d.get("market_cap", 0))), 2),
            "Volume (Lakhs)":   round(vol / 1e5, 2),
            "Value (₹ Crores)": round(tv_raw / 1e7, 2),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values("Value (₹ Crores)", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.index += 1
    return df


def normalise_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, thousands=",")
    except Exception as e:
        print(f"  CSV read error: {e}")
        return pd.DataFrame()

    df.columns = df.columns.str.strip()
    print(f"  CSV columns: {list(df.columns)}")

    col_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if cl == "symbol":
            col_map[col] = "Symbol"
        elif cl == "series":
            col_map[col] = "Series"
        elif cl in ("ltp", "last price", "close", "lastprice"):
            col_map[col] = "LTP (₹)"
        elif cl in ("%chng", "%change", "% change", "pchange",
                    "% chng", "per change", "%chg"):
            col_map[col] = "% Change"
        elif "mkt cap" in cl or "market cap" in cl:
            col_map[col] = "Mkt Cap (₹ Cr)"
        elif "volume" in cl:
            col_map[col] = "Volume (Lakhs)"
        elif "value" in cl:
            col_map[col] = "Value (₹ Crores)"
        elif "company" in cl or cl == "name":
            col_map[col] = "Company"

    df.rename(columns=col_map, inplace=True)

    for col in ["LTP (₹)", "% Change", "Mkt Cap (₹ Cr)",
                "Volume (Lakhs)", "Value (₹ Crores)"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "", regex=False).str.strip(),
                errors="coerce"
            ).fillna(0)

    for col, default in [("Company", ""), ("Series", "EQ"),
                         ("Mkt Cap (₹ Cr)", 0.0)]:
        if col not in df.columns:
            df[col] = default

    if "Value (₹ Crores)" in df.columns:
        df.sort_values("Value (₹ Crores)", ascending=False, inplace=True)

    df.reset_index(drop=True, inplace=True)
    df.index += 1
    print(f"  {len(df)} rows loaded from CSV")
    return df


# ═══════════════════════════════════════════════════════════════
#  STEP 1 — MASTER DOWNLOAD  (from nse_pipeline.py)
# ═══════════════════════════════════════════════════════════════

def download_nse_data(headless: bool, from_csv: str) -> pd.DataFrame:
    if from_csv:
        print(f"\n  Loading manual CSV: {from_csv}")
        df = normalise_csv(from_csv)
        if df.empty:
            print("  [ERROR] CSV empty or unreadable.")
            sys.exit(1)
        return df

    if not SELENIUM_OK:
        print("[ERROR] selenium not installed.")
        sys.exit(1)

    dl_dir = tempfile.mkdtemp()
    driver = build_driver(headless, dl_dir)
    driver.set_page_load_timeout(60)
    patch_driver(driver)
    df = pd.DataFrame()

    try:
        warm_session(driver)
        records = fetch_via_xhr(driver)
        if records:
            df = normalise_json(records)
        if df.empty:
            print("\n  XHR returned no data — trying CSV button …")
            csv_path = fetch_via_csv_button(driver, dl_dir)
            if csv_path:
                df = normalise_csv(csv_path)
    except WebDriverException as e:
        print(f"\n[ERROR] WebDriver: {e}")
    finally:
        driver.quit()
        shutil.rmtree(dl_dir, ignore_errors=True)
        print("  Browser closed.")

    if df.empty:
        print("\n  ✗  Could not retrieve data from NSE.")
        print(f"  MANUAL FALLBACK:")
        print(f"  1. Open {NSE_PAGE} in Chrome")
        print("  2. Click the ↓ CSV button")
        print("  3. Run: python chart_generator_4tf_nse.py --from-csv StocksTraded.csv")
        sys.exit(1)

    return df


# ═══════════════════════════════════════════════════════════════
#  STEP 2 — FILTER  (from nse_pipeline.py)
# ═══════════════════════════════════════════════════════════════

def filter_stocks(df: pd.DataFrame, min_value_cr: float) -> pd.DataFrame:
    print(f"\n  Top 5 by Value before filter:")
    top5 = df[df["Value (₹ Crores)"] > 0].nlargest(5, "Value (₹ Crores)")
    for _, r in top5.iterrows():
        print(f"    {r['Symbol']:<12}  ₹{r['Value (₹ Crores)']:>10,.2f} Cr"
              f"  Series={r['Series']}")

    mask = (
        (df["Series"].str.strip().str.upper() == "EQ") &
        (df["Value (₹ Crores)"] > min_value_cr)
    )
    out = df[mask].copy()
    out.sort_values("Value (₹ Crores)", ascending=False, inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


# ═══════════════════════════════════════════════════════════════
#  STEP 3 — CHART HELPERS  (from chart_generator_4tf.py)
# ═══════════════════════════════════════════════════════════════

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
    end_dt   = datetime.datetime.today() + timedelta(days=1)
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

        if interval == "1h":
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            df.index = df.index.tz_convert("Asia/Kolkata")
            df = df.between_time("09:15", "15:30")
            df.index = df.index.tz_localize(None)

        min_bars = 1 if interval == "1mo" else MACD_SLOW + MACD_SIGNAL + 2
        if len(df) < min_bars:
            return None

        return df.tail(N_BARS)

    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════
#  STEP 3 — DRAW ONE QUADRANT  (from chart_generator_4tf.py)
# ═══════════════════════════════════════════════════════════════

def draw_quarter(ax_price, ax_macd, df, label, colors, date_fmt, y_side="right"):
    s  = STYLE
    n  = len(df)
    xs = np.arange(n)

    ema9 = ema(df["Close"], EMA_PERIOD)

    has_macd = n >= MACD_SLOW + MACD_SIGNAL + 2
    if has_macd:
        macd_l, sig, hist = macd_calc(df["Close"])
        hist_vals = hist.values
        macd_bull_turn = np.zeros(n, dtype=bool)
        macd_bear_turn = np.zeros(n, dtype=bool)
        for i in range(1, n):
            if hist_vals[i - 1] <= 0 and hist_vals[i] > 0:
                macd_bull_turn[i] = True
            if hist_vals[i - 1] >= 0 and hist_vals[i] < 0:
                macd_bear_turn[i] = True
    else:
        macd_l = sig = hist = hist_vals = None
        macd_bull_turn = macd_bear_turn = np.zeros(n, dtype=bool)

    # ── Style axes ───────────────────────────────────────────────────────────
    for ax in (ax_price, ax_macd):
        ax.set_facecolor(s["panel_bg"])
        ax.tick_params(colors=s["text"], labelsize=7, width=1.0, length=3)
        for spine in ax.spines.values():
            spine.set_edgecolor(s["border"])
            spine.set_linewidth(0.8)
        ax.grid(True, color=s["grid"], linewidth=0.5, alpha=0.9)

    # ── Candlesticks: vlines wick + Rectangle body ────────────────────────────
    avg_range  = (df["High"] - df["Low"]).mean()
    min_body_h = avg_range * 0.04
    BODY_W     = 0.6

    opens  = df["Open"].values
    closes = df["Close"].values
    highs  = df["High"].values
    lows   = df["Low"].values

    bull_mask  = closes >= opens
    bear_mask  = ~bull_mask
    body_tops  = np.maximum(opens, closes)
    body_bots  = np.minimum(opens, closes)
    doji       = (body_tops - body_bots) < min_body_h
    body_tops  = np.where(doji, body_bots + min_body_h, body_tops)

    # Wicks first (thin, same colour as body, exactly at xs[i])
    if bull_mask.any():
        ax_price.vlines(xs[bull_mask], lows[bull_mask], highs[bull_mask],
                        color=s["up"], linewidth=1.0, zorder=2)
    if bear_mask.any():
        ax_price.vlines(xs[bear_mask], lows[bear_mask], highs[bear_mask],
                        color=s["dn"], linewidth=1.0, zorder=2)

    # Bodies — Rectangle centred exactly at xs[i] (no bar() alignment drift)
    for i in range(n):
        bc = s["up"] if bull_mask[i] else s["dn"]
        ax_price.add_patch(Rectangle(
            (xs[i] - BODY_W / 2, body_bots[i]),
            BODY_W, body_tops[i] - body_bots[i],
            facecolor=bc, edgecolor=bc, linewidth=0, zorder=3,
        ))

    # ── EMA ───────────────────────────────────────────────────────────────────
    ax_price.plot(xs, ema9.values, color=colors["ema"],
                  linewidth=1.8, zorder=4, label=f"EMA {EMA_PERIOD}")

    # ── MACD turn circles on price panel ─────────────────────────────────────
    price_range = df["High"].max() - df["Low"].min()
    circle_off  = price_range * 0.025
    for i in range(n):
        if macd_bull_turn[i]:
            ax_price.plot(xs[i], lows[i] - circle_off,
                          marker="o", markersize=8,
                          color=s["macd_turn_bull"], markeredgecolor="white",
                          markeredgewidth=0.8, zorder=7, linestyle="None")
        if macd_bear_turn[i]:
            ax_price.plot(xs[i], highs[i] + circle_off,
                          marker="o", markersize=8,
                          color=s["macd_turn_bear"], markeredgecolor="white",
                          markeredgewidth=0.8, zorder=7, linestyle="None")

    # ── Axes limits ───────────────────────────────────────────────────────────
    ax_price.set_xlim(-1, n + 1)
    pad = price_range * 0.07
    ax_price.set_ylim(df["Low"].min() - pad, df["High"].max() + pad)
    ax_price.yaxis.set_label_position(y_side)
    ax_price.yaxis.tick_right() if y_side == "right" else ax_price.yaxis.tick_left()
    ax_price.set_ylabel("Price (₹)", color=s["text"], fontsize=7)
    ax_price.yaxis.set_tick_params(labelsize=7, labelcolor=s["text"])
    ax_macd.yaxis.set_label_position(y_side)
    ax_macd.yaxis.tick_right() if y_side == "right" else ax_macd.yaxis.tick_left()
    ax_macd.set_ylabel("MACD", color=s["text"], fontsize=7)
    ax_macd.yaxis.set_tick_params(labelsize=6, labelcolor=s["text"])

    # ── Price pills ───────────────────────────────────────────────────────────
    last_close = df["Close"].iloc[-1]
    last_ema   = ema9.iloc[-1]
    close_col  = s["up"] if last_close >= df["Open"].iloc[-1] else s["dn"]
    for val, col in [(last_close, close_col), (last_ema, colors["ema"])]:
        ax_price.annotate(f"₹{val:,.2f}",
                          xy=(1, val), xycoords=("axes fraction", "data"),
                          xytext=(4, 0), textcoords="offset points",
                          fontsize=7.5, fontweight="bold", color="#FFFFFF",
                          ha="left", va="center", annotation_clip=False,
                          bbox=dict(boxstyle="round,pad=0.3",
                                    facecolor=col, edgecolor="none", alpha=0.97))

    # ── Legend ────────────────────────────────────────────────────────────────
    leg = [
        mpatches.Patch(facecolor=s["up"], label="Bullish"),
        mpatches.Patch(facecolor=s["dn"], label="Bearish"),
        Line2D([0],[0], color=colors["ema"], lw=1.5, label=f"EMA {EMA_PERIOD}"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=s["macd_turn_bull"],
               ms=7, ls="None", label="MACD turns +ve"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=s["macd_turn_bear"],
               ms=7, ls="None", label="MACD turns -ve"),
    ]
    ax_price.legend(handles=leg, loc="upper left", fontsize=6.5,
                    framealpha=0.85, facecolor=s["bg"],
                    edgecolor=s["border"], labelcolor=s["text"])

    # ── Timeframe badge ───────────────────────────────────────────────────────
    ax_price.set_title(f"  {label}  ", loc="left", color="#FFFFFF",
                       fontsize=10, fontweight="bold", pad=3,
                       bbox=dict(boxstyle="round,pad=0.35",
                                 facecolor=colors["ema"],
                                 edgecolor="none", alpha=0.95))

    # ── MACD panel ────────────────────────────────────────────────────────────
    if has_macd:
        hcols = [colors["hu"] if v >= 0 else colors["hd"] for v in hist_vals]
        ax_macd.bar(xs, hist_vals, color=hcols, alpha=0.80, width=0.65, zorder=2)
        ax_macd.plot(xs, macd_l.values, color=colors["macd"], lw=1.0,
                     zorder=3, label="MACD")
        ax_macd.plot(xs, sig.values,    color=colors["sig"],  lw=0.9,
                     zorder=3, label="Signal")
        ax_macd.axhline(0, color=s["zero_line"], lw=0.8, ls="--")
        ax_macd.legend(loc="upper left", fontsize=6.5, framealpha=0.85,
                       facecolor=s["bg"], edgecolor=s["border"],
                       labelcolor=s["text"])
    else:
        ax_macd.set_facecolor(s["panel_bg"])
        ax_macd.set_xticks([]); ax_macd.set_yticks([])
        ax_macd.text(0.5, 0.5,
                     f"MACD needs ≥{MACD_SLOW + MACD_SIGNAL + 2} bars  (have {n})",
                     transform=ax_macd.transAxes, color=s["subtext"],
                     fontsize=7, ha="center", va="center", style="italic")

    # ── X-axis labels ─────────────────────────────────────────────────────────
    step = max(n // 8, 1)
    ax_macd.set_xticks(xs[::step])
    ax_macd.set_xticklabels(
        [df.index[i].strftime(date_fmt) for i in range(0, n, step)],
        rotation=25, ha="right", fontsize=7, color=s["text"])
    plt.setp(ax_price.get_xticklabels(), visible=False)

    # ── % change caption ──────────────────────────────────────────────────────
    lc  = df["Close"].iloc[-1]; fc = df["Close"].iloc[0]
    pct = (lc - fc) / fc * 100
    ax_price.text(0.99, 0.02, f"{'+' if pct>=0 else ''}{pct:.2f}%  ({n} bars)",
                  transform=ax_price.transAxes,
                  color=s["up"] if pct >= 0 else s["dn"],
                  fontsize=7.5, fontweight="bold", ha="right", va="bottom")


# ═══════════════════════════════════════════════════════════════
#  STEP 3 — 4-QUADRANT MASTER CHART  (from chart_generator_4tf.py)
# ═══════════════════════════════════════════════════════════════

def plot_chart(symbol, d_df, w_df, m_df, h_df, output_path):
    s = STYLE

    fig = plt.figure(figsize=(19.2, 10.4), dpi=100, facecolor=s["bg"])
    fig.patch.set_facecolor(s["bg"])

    gs = gridspec.GridSpec(4, 2,
                           height_ratios=[7, 3, 7, 3],
                           width_ratios=[1, 1],
                           hspace=0.10, wspace=0.14,
                           top=0.93, bottom=0.05,
                           left=0.02, right=0.97)

    ax_d_p  = fig.add_subplot(gs[0, 0])
    ax_d_m  = fig.add_subplot(gs[1, 0], sharex=ax_d_p)
    ax_w_p  = fig.add_subplot(gs[0, 1])
    ax_w_m  = fig.add_subplot(gs[1, 1], sharex=ax_w_p)
    ax_mo_p = fig.add_subplot(gs[2, 0])
    ax_mo_m = fig.add_subplot(gs[3, 0], sharex=ax_mo_p)
    ax_h_p  = fig.add_subplot(gs[2, 1])
    ax_h_m  = fig.add_subplot(gs[3, 1], sharex=ax_h_p)

    tf_colors = {
        "d":  dict(up=s["up"], dn=s["dn"], ema=s["d_ema"],
                   macd=s["d_macd"], sig=s["d_sig"], hu=s["d_hu"], hd=s["d_hd"]),
        "w":  dict(up=s["up"], dn=s["dn"], ema=s["w_ema"],
                   macd=s["w_macd"], sig=s["w_sig"], hu=s["w_hu"], hd=s["w_hd"]),
        "mo": dict(up=s["up"], dn=s["dn"], ema=s["m_ema"],
                   macd=s["m_macd"], sig=s["m_sig"], hu=s["m_hu"], hd=s["m_hd"]),
        "h":  dict(up=s["up"], dn=s["dn"], ema=s["h_ema"],
                   macd=s["h_macd"], sig=s["h_sig"], hu=s["h_hu"], hd=s["h_hd"]),
    }

    def no_data(ax_p, ax_m, label, key):
        col = tf_colors[key]["ema"]
        for ax in (ax_p, ax_m):
            ax.set_facecolor(s["panel_bg"])
            for spine in ax.spines.values():
                spine.set_edgecolor(s["border"])
            ax.set_xticks([]); ax.set_yticks([])
            ax.tick_params(left=False, right=False, bottom=False,
                           labelbottom=False, labelleft=False, labelright=False)
        ax_p.set_title(f"  {label}  ", loc="left", color="#FFFFFF",
                       fontsize=10, fontweight="bold", pad=3,
                       bbox=dict(boxstyle="round,pad=0.35",
                                 facecolor=col, edgecolor="none", alpha=0.95))
        ax_p.text(0.5, 0.5, "No data\n(recently listed stock)",
                  transform=ax_p.transAxes, color=s["subtext"],
                  fontsize=10, ha="center", va="center", style="italic")
        ax_m.text(0.5, 0.5, "—", transform=ax_m.transAxes,
                  color=s["subtext"], fontsize=10, ha="center", va="center")

    quarters = [
        (ax_d_p,  ax_d_m,  d_df,  "Daily",   "d",  "%d %b '%y",   "left"),
        (ax_w_p,  ax_w_m,  w_df,  "Weekly",  "w",  "%d %b '%y",   "right"),
        (ax_mo_p, ax_mo_m, m_df,  "Monthly", "mo", "%b '%y",      "left"),
        (ax_h_p,  ax_h_m,  h_df,  "Hourly",  "h",  "%d %b %H:%M", "right"),
    ]

    for ax_p, ax_m, df, label, key, dfmt, yside in quarters:
        if df is not None and len(df) >= 1:
            draw_quarter(ax_p, ax_m, df, label, tf_colors[key], dfmt, yside)
        else:
            no_data(ax_p, ax_m, label, key)

    # ── Master title ──────────────────────────────────────────────────────────
    latest = (d_df.index[-1].strftime("%d %b %Y")
              if d_df is not None and not d_df.empty else "–")
    lc = (d_df["Close"].iloc[-1] if d_df is not None and not d_df.empty else 0)

    fig.text(0.03, 0.965,
             f"{symbol}  |  NSE  |  Multi-Timeframe Chart",
             color=s["text"], fontsize=12, fontweight="bold")
    fig.text(0.03, 0.945,
             f"₹{lc:,.2f}  |  {N_BARS} bars per timeframe"
             f"  |  ● MACD turns +ve   ● MACD turns -ve",
             color=s["subtext"], fontsize=8)
    fig.text(0.97, 0.965, f"Latest: {latest}",
             color=s["text"], fontsize=8, ha="right", fontweight="bold")
    fig.text(0.97, 0.945,
             f"EMA {EMA_PERIOD}  |  MACD ({MACD_FAST},{MACD_SLOW},{MACD_SIGNAL})"
             f"  |  Data: yfinance",
             color=s["subtext"], fontsize=7, ha="right")

    fig.add_artist(plt.Line2D([0.02, 0.97], [0.50, 0.50],
                              transform=fig.transFigure,
                              color=s["border"], linewidth=1.5, alpha=0.8))
    fig.add_artist(plt.Line2D([0.50, 0.50], [0.05, 0.93],
                              transform=fig.transFigure,
                              color=s["border"], linewidth=1.5, alpha=0.8))

    plt.savefig(output_path, dpi=100, facecolor=s["bg"], edgecolor="none")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
#  STEP 3 — BATCH CHART GENERATOR
# ═══════════════════════════════════════════════════════════════

def generate_charts(filtered_df: pd.DataFrame):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    success, failed = [], []
    total = len(filtered_df)

    for idx, row in enumerate(filtered_df.itertuples(), 1):
        sym    = row.Symbol
        ticker = sym if sym.endswith(EXCHANGE_SFX) else sym + EXCHANGE_SFX

        try:
            tv_cr = filtered_df.iloc[idx - 1]["Value (₹ Crores)"]
        except Exception:
            tv_cr = 0.0

        print(f"\n[{idx:>4}/{total}]  {ticker:<22}  "
              f"₹{tv_cr:>8,.1f} Cr", end="  ", flush=True)

        try:
            d_df = fetch_ohlc(ticker, "1d")
            w_df = fetch_ohlc(ticker, "1wk")
            m_df = fetch_ohlc(ticker, "1mo")
            h_df = fetch_ohlc(ticker, "1h")

            counts = (f"D:{len(d_df) if d_df is not None else 0}"
                      f"  W:{len(w_df) if w_df is not None else 0}"
                      f"  M:{len(m_df) if m_df is not None else 0}"
                      f"  H:{len(h_df) if h_df is not None else 0}")
            print(counts, end="  ", flush=True)

            out = os.path.join(OUTPUT_DIR, f"{sym}.png")
            plot_chart(sym, d_df, w_df, m_df, h_df, out)
            print(f"→  {out}")
            success.append(sym)

        except Exception:
            print("✗  Error")
            traceback.print_exc()
            failed.append(sym)

    return success, failed


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="NSE Live → Filter → 4-Timeframe Charts"
    )
    parser.add_argument("--visible",   action="store_true",
                        help="Run Chrome with a visible window (local debug)")
    parser.add_argument("--from-csv",  metavar="FILE",
                        help="Skip browser — parse a manually downloaded NSE CSV")
    parser.add_argument("--min-value", type=float, default=TRADED_VALUE_MIN_CR,
                        help=f"Min traded value in ₹ Cr (default: {TRADED_VALUE_MIN_CR})")
    args     = parser.parse_args()
    headless = not args.visible

    run_time = datetime.datetime.now().strftime("%d %b %Y  %H:%M:%S")
    print(f"\n{'═'*65}")
    print(f"  4-TF Chart Generator  —  NSE Live  |  {run_time}")
    print(f"  Step 1 : Download NSE Stocks Traded (live)")
    print(f"  Step 2 : Filter  EQ series, Value > ₹{args.min_value} Cr")
    print(f"  Step 3 : 4-TF charts (Daily/Weekly/Monthly/Hourly)  →  {OUTPUT_DIR}/")
    print(f"{'═'*65}")

    # ── STEP 1 ───────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  STEP 1  —  NSE live data download")
    print(f"{'─'*65}")
    all_df = download_nse_data(
        headless=headless,
        from_csv=getattr(args, "from_csv", None),
    )
    print(f"\n  Total records : {len(all_df)}")
    all_df.to_csv("nse_all_stocks.csv", index=False)
    print(f"  Saved         : nse_all_stocks.csv")

    # ── STEP 2 ───────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  STEP 2  —  Filter: Value > ₹{args.min_value} Cr  (EQ only)")
    print(f"{'─'*65}")
    filtered = filter_stocks(all_df, args.min_value)
    print(f"\n  Stocks passing filter : {len(filtered)}")

    if filtered.empty:
        print("\n  ⚠  No stocks passed the filter.")
        print("     Inspect nse_all_stocks.csv — check 'Value (₹ Crores)'.")
        sys.exit(0)

    print(f"\n  {'#':>4}  {'Symbol':<12} {'Company':<28} "
          f"{'LTP':>8}  {'Value (Cr)':>11}  {'%Chg':>7}")
    print(f"  {'─'*78}")
    for i, row in filtered.head(25).iterrows():
        print(f"  {i+1:>4}  {row['Symbol']:<12} "
              f"{str(row.get('Company',''))[:26]:<28} "
              f"  ₹{row['LTP (₹)']:>7,.2f}"
              f"  ₹{row['Value (₹ Crores)']:>9,.1f} Cr"
              f"  {row['% Change']:>+7.2f}%")
    if len(filtered) > 25:
        print(f"  … and {len(filtered) - 25} more stocks")

    filtered.to_csv("nse_filtered_stocks.csv", index=False)
    print(f"\n  Saved : nse_filtered_stocks.csv  ({len(filtered)} stocks)")

    # ── STEP 3 ───────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  STEP 3  —  Generating 4-TF charts  →  {OUTPUT_DIR}/")
    print(f"  {N_BARS} bars per timeframe  |  White bg  |  Green/Red candles")
    print(f"{'─'*65}")
    success, failed = generate_charts(filtered)

    print(f"\n{'═'*65}")
    print(f"  DONE  —  {run_time}")
    print(f"  NSE records  : {len(all_df)}")
    print(f"  Filtered     : {len(filtered)}  (Value > ₹{args.min_value} Cr)")
    print(f"  Charts saved : {len(success)}  →  {OUTPUT_DIR}/")
    if failed:
        print(f"  Failed       : {len(failed)}: "
              + ", ".join(failed[:20])
              + (" …" if len(failed) > 20 else ""))
    print(f"  nse_all_stocks.csv      — full NSE list")
    print(f"  nse_filtered_stocks.csv — filtered list")
    print(f"{'═'*65}\n")


if __name__ == "__main__":
    main()
