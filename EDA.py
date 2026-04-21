import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =========================================================
# CONFIG
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "stock_data.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
SCREENSHOT_DIR = BASE_DIR / "screenshots"

OUTPUT_DIR.mkdir(exist_ok=True)
SCREENSHOT_DIR.mkdir(exist_ok=True)

initial_cash = 100000

# Outlier cleaning thresholds for market-intelligence use
RETURN_CLIP_LOWER = -0.20
RETURN_CLIP_UPPER = 0.20


# =========================================================
# HELPERS
# =========================================================
def print_section(title: str):
    print(f"\n{'=' * 70}")
    print(title)
    print(f"{'=' * 70}")


def find_col(df: pd.DataFrame, candidates):
    if df.empty:
        return None

    cols = list(df.columns)
    lower_map = {str(c).lower(): c for c in cols}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    for candidate in candidates:
        for col in cols:
            if candidate.lower() in str(col).lower():
                return col

    return None


def safe_numeric(df: pd.DataFrame, cols):
    df = df.copy()
    for col in cols:
        if col and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def save_text_summary(lines, file_path: Path):
    pd.DataFrame({"summary_line": lines}).to_csv(file_path, index=False)


def classify_market_regime(std_return, outlier_share):
    if pd.isna(std_return):
        return "Unknown"

    if std_return < 0.02 and outlier_share < 0.01:
        return "Low-volatility stable market"
    if std_return < 0.05 and outlier_share < 0.03:
        return "Moderate-volatility controlled market"
    return "High-volatility / event-driven market"


# =========================================================
# LOAD DATA
# =========================================================
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print_section("FIRST 5 ROWS")
print(df.head())

print_section("DATA INFO")
df.info()

print_section("MISSING VALUES")
print(df.isnull().sum())

print_section("DUPLICATES")
print(df.duplicated().sum())


# =========================================================
# COLUMN DETECTION
# =========================================================
ticker_col = find_col(df, ["ticker_clean", "ticker", "symbol", "stock", "security"])
date_col = find_col(df, ["date", "datetime"])
close_col = find_col(df, ["close", "closing_price", "adj_close", "price"])
open_col = find_col(df, ["open"])
high_col = find_col(df, ["high"])
low_col = find_col(df, ["low"])
volume_col = find_col(df, ["volume", "vol"])

print_section("DETECTED COLUMNS")
print(
    {
        "ticker_col": ticker_col,
        "date_col": date_col,
        "close_col": close_col,
        "open_col": open_col,
        "high_col": high_col,
        "low_col": low_col,
        "volume_col": volume_col,
    }
)

if ticker_col is None:
    raise ValueError("No ticker column found.")
if date_col is None:
    raise ValueError("No date column found.")
if close_col is None:
    raise ValueError("No close price column found.")


# =========================================================
# CLEAN DATA
# =========================================================
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col]).copy()

df = safe_numeric(df, [open_col, high_col, low_col, close_col, volume_col])

df = df.sort_values([ticker_col, date_col]).reset_index(drop=True)

print_section("DATE RANGE")
print("Start:", df[date_col].min())
print("End:", df[date_col].max())

print_section("BASIC STATS")
print(df.describe(include="all"))

print_section("UNIQUE STOCKS")
print(df[ticker_col].nunique())

print_section("SAMPLE TICKERS")
print(df[ticker_col].dropna().astype(str).unique()[:10])


# =========================================================
# FEATURE ENGINEERING FOR EDA
# =========================================================
df["daily_return_raw"] = df.groupby(ticker_col)[close_col].pct_change()
df["daily_return_raw"] = df["daily_return_raw"].replace([np.inf, -np.inf], np.nan)

# Cleaned return for dashboard-facing market intelligence
df["daily_return_clean"] = df["daily_return_raw"].clip(RETURN_CLIP_LOWER, RETURN_CLIP_UPPER)

df["price_change"] = df.groupby(ticker_col)[close_col].diff()
df["rolling_30d_volatility_raw"] = df.groupby(ticker_col)["daily_return_raw"].transform(
    lambda s: s.rolling(30, min_periods=5).std()
)
df["rolling_30d_volatility_clean"] = df.groupby(ticker_col)["daily_return_clean"].transform(
    lambda s: s.rolling(30, min_periods=5).std()
)

if volume_col:
    df["volume_change"] = df.groupby(ticker_col)[volume_col].pct_change()
else:
    df["volume_change"] = np.nan

clean_returns_raw = df["daily_return_raw"].dropna().copy()
clean_returns_cleaned = df["daily_return_clean"].dropna().copy()


# =========================================================
# DATASET SUMMARY EXPORT
# =========================================================
dataset_summary = pd.DataFrame(
    {
        "metric": [
            "row_count",
            "column_count",
            "unique_tickers",
            "start_date",
            "end_date",
            "missing_close_values",
            "duplicate_rows",
            "return_clip_lower",
            "return_clip_upper",
        ],
        "value": [
            len(df),
            len(df.columns),
            df[ticker_col].nunique(),
            df[date_col].min(),
            df[date_col].max(),
            df[close_col].isna().sum(),
            df.duplicated().sum(),
            RETURN_CLIP_LOWER,
            RETURN_CLIP_UPPER,
        ],
    }
)
dataset_summary.to_csv(OUTPUT_DIR / "eda_dataset_summary.csv", index=False)


# =========================================================
# RAW MARKET BEHAVIOUR SUMMARY
# =========================================================
raw_mean_return = clean_returns_raw.mean()
raw_median_return = clean_returns_raw.median()
raw_std_return = clean_returns_raw.std()
raw_min_return = clean_returns_raw.min()
raw_max_return = clean_returns_raw.max()

raw_outlier_mask = clean_returns_raw.abs() > 0.10
raw_outlier_count = int(raw_outlier_mask.sum())
raw_outlier_share = float(raw_outlier_count / len(clean_returns_raw)) if len(clean_returns_raw) > 0 else np.nan

raw_market_regime = classify_market_regime(raw_std_return, raw_outlier_share)

raw_market_behaviour_summary = pd.DataFrame(
    [
        {"metric": "mean_daily_return_raw", "value": raw_mean_return},
        {"metric": "median_daily_return_raw", "value": raw_median_return},
        {"metric": "daily_return_std_raw", "value": raw_std_return},
        {"metric": "min_daily_return_raw", "value": raw_min_return},
        {"metric": "max_daily_return_raw", "value": raw_max_return},
        {"metric": "outlier_count_abs_gt_10pct_raw", "value": raw_outlier_count},
        {"metric": "outlier_share_raw", "value": raw_outlier_share},
        {"metric": "market_regime_raw", "value": raw_market_regime},
    ]
)
raw_market_behaviour_summary.to_csv(OUTPUT_DIR / "eda_market_behaviour_summary_raw.csv", index=False)


# =========================================================
# CLEANED MARKET BEHAVIOUR SUMMARY
# =========================================================
clean_mean_return = clean_returns_cleaned.mean()
clean_median_return = clean_returns_cleaned.median()
clean_std_return = clean_returns_cleaned.std()
clean_min_return = clean_returns_cleaned.min()
clean_max_return = clean_returns_cleaned.max()

clean_outlier_mask = clean_returns_cleaned.abs() > 0.10
clean_outlier_count = int(clean_outlier_mask.sum())
clean_outlier_share = float(clean_outlier_count / len(clean_returns_cleaned)) if len(clean_returns_cleaned) > 0 else np.nan

clean_market_regime = classify_market_regime(clean_std_return, clean_outlier_share)

market_behaviour_summary = pd.DataFrame(
    [
        {"metric": "mean_daily_return", "value": clean_mean_return},
        {"metric": "median_daily_return", "value": clean_median_return},
        {"metric": "daily_return_std", "value": clean_std_return},
        {"metric": "min_daily_return", "value": clean_min_return},
        {"metric": "max_daily_return", "value": clean_max_return},
        {"metric": "outlier_count_abs_gt_10pct", "value": clean_outlier_count},
        {"metric": "outlier_share", "value": clean_outlier_share},
        {"metric": "market_regime", "value": clean_market_regime},
    ]
)
market_behaviour_summary.to_csv(OUTPUT_DIR / "eda_market_behaviour_summary.csv", index=False)


# =========================================================
# NOTEBOOK-STYLE INTERPRETATION EXPORT
# =========================================================
distribution_comment = (
    "Daily returns are heavily concentrated around zero, suggesting that most trading days see only small price changes."
)

if abs(clean_mean_return) < 0.001:
    bias_comment = "Mean daily return is near zero, indicating a broadly neutral drift over time."
elif clean_mean_return > 0:
    bias_comment = "Mean daily return is positive, suggesting a slight upward bias over time."
else:
    bias_comment = "Mean daily return is negative, suggesting a slight downward bias over time."

if pd.notna(clean_std_return) and clean_std_return < 0.02:
    volatility_comment = "Volatility is low, indicating a relatively stable and slow-moving market."
elif pd.notna(clean_std_return) and clean_std_return < 0.05:
    volatility_comment = "Volatility is moderate, indicating controlled price movement rather than chaotic swings."
else:
    volatility_comment = "Volatility is high, indicating stronger market swings and higher trading risk."

clipped_observation_count = int((df["daily_return_raw"] != df["daily_return_clean"]).fillna(False).sum())

if clipped_observation_count > 0:
    outlier_comment = (
        f"Extreme return values were detected and cleaned ({clipped_observation_count} observations clipped outside {RETURN_CLIP_LOWER:.0%} to {RETURN_CLIP_UPPER:.0%}), reducing distortion from anomalies or structural adjustment effects."
    )
else:
    outlier_comment = "No major return outliers required clipping under the current cleaning rule."

if pd.notna(clean_std_return) and clean_std_return < 0.05 and abs(clean_mean_return) < 0.01:
    core_insight = (
        "The NGX market shows clustering around small returns with occasional sharp moves, suggesting a slower-moving market with intermittent bursts of opportunity."
    )
else:
    core_insight = (
        "The NGX market shows meaningful day-to-day movement with event-driven bursts, suggesting a market where timing and risk control matter strongly."
    )

trading_favors = []
trading_avoid = []

if pd.notna(clean_std_return) and clean_std_return < 0.05:
    trading_favors.extend(
        [
            "Momentum strategies",
            "Breakout trading around catalyst periods",
            "Event-driven entries",
            "Institutional flow tracking",
        ]
    )
    trading_avoid.extend(
        [
            "High-frequency trading",
            "Scalping strategies",
        ]
    )
else:
    trading_favors.extend(
        [
            "Volatility breakout strategies",
            "Shorter-term swing trading",
            "Risk-managed event trading",
        ]
    )
    trading_avoid.extend(
        [
            "Overleveraged position sizing",
            "Blind trend following without risk filters",
        ]
    )

market_interpretation = pd.DataFrame(
    [
        {"section": "distribution", "insight": distribution_comment},
        {"section": "central_tendency", "insight": bias_comment},
        {"section": "volatility", "insight": volatility_comment},
        {"section": "outliers", "insight": outlier_comment},
        {"section": "core_insight", "insight": core_insight},
    ]
)
market_interpretation.to_csv(OUTPUT_DIR / "eda_market_interpretation.csv", index=False)

trading_implications = pd.DataFrame(
    [{"category": "favors", "strategy": item} for item in trading_favors]
    + [{"category": "avoid", "strategy": item} for item in trading_avoid]
)
trading_implications.to_csv(OUTPUT_DIR / "eda_trading_implications.csv", index=False)


# =========================================================
# LIQUIDITY / MOST ACTIVE STOCKS
# =========================================================
liquidity_summary = (
    df[ticker_col]
    .astype(str)
    .value_counts()
    .reset_index()
)
liquidity_summary.columns = [ticker_col, "record_count"]
liquidity_summary = liquidity_summary.sort_values("record_count", ascending=False)

top_active_stocks = liquidity_summary.head(10).copy()
top_active_stocks.to_csv(OUTPUT_DIR / "eda_top_active_stocks.csv", index=False)
liquidity_summary.to_csv(OUTPUT_DIR / "eda_liquidity_summary.csv", index=False)


# =========================================================
# TICKER-LEVEL SUMMARY
# =========================================================
ticker_summary = (
    df.groupby(ticker_col)
    .agg(
        observation_count=(close_col, "count"),
        first_date=(date_col, "min"),
        last_date=(date_col, "max"),
        latest_close=(close_col, "last"),
        avg_close=(close_col, "mean"),
        min_close=(close_col, "min"),
        max_close=(close_col, "max"),
        avg_daily_return_raw=("daily_return_raw", "mean"),
        avg_daily_return_clean=("daily_return_clean", "mean"),
        daily_return_std_raw=("daily_return_raw", "std"),
        daily_return_std_clean=("daily_return_clean", "std"),
        latest_30d_volatility_raw=("rolling_30d_volatility_raw", "last"),
        latest_30d_volatility_clean=("rolling_30d_volatility_clean", "last"),
    )
    .reset_index()
    .sort_values("avg_daily_return_clean", ascending=False)
)
ticker_summary.to_csv(OUTPUT_DIR / "eda_ticker_summary.csv", index=False)


# =========================================================
# TOP GAINERS / LOSERS
# =========================================================
latest_per_stock = (
    df.sort_values([ticker_col, date_col])
    .groupby(ticker_col, as_index=False)
    .tail(1)
    .copy()
)

top_gainers = (
    latest_per_stock[[ticker_col, date_col, close_col, "daily_return_raw", "daily_return_clean", "rolling_30d_volatility_clean"]]
    .sort_values("daily_return_clean", ascending=False)
    .head(10)
)

top_losers = (
    latest_per_stock[[ticker_col, date_col, close_col, "daily_return_raw", "daily_return_clean", "rolling_30d_volatility_clean"]]
    .sort_values("daily_return_clean", ascending=True)
    .head(10)
)

top_gainers.to_csv(OUTPUT_DIR / "eda_top_gainers.csv", index=False)
top_losers.to_csv(OUTPUT_DIR / "eda_top_losers.csv", index=False)


# =========================================================
# TEXT SUMMARIES FOR DASHBOARD / AI
# =========================================================
market_summary_lines = [
    f"Market regime: {clean_market_regime}",
    f"Mean daily return: {clean_mean_return:.6f}",
    f"Median daily return: {clean_median_return:.6f}",
    f"Daily return volatility: {clean_std_return:.6f}",
    f"Minimum daily return: {clean_min_return:.6f}",
    f"Maximum daily return: {clean_max_return:.6f}",
    f"Outlier count (>10% absolute move): {clean_outlier_count}",
    f"Outlier share: {clean_outlier_share:.4%}" if pd.notna(clean_outlier_share) else "Outlier share: N/A",
    f"Clipped extreme observations: {clipped_observation_count}",
]
save_text_summary(market_summary_lines, OUTPUT_DIR / "eda_market_summary_lines.csv")

interpretation_lines = [
    distribution_comment,
    bias_comment,
    volatility_comment,
    outlier_comment,
    core_insight,
]
save_text_summary(interpretation_lines, OUTPUT_DIR / "eda_interpretation_lines.csv")


# =========================================================
# CHARTS
# =========================================================
# 1. Raw distribution of daily returns
plt.figure(figsize=(10, 5))
clean_returns_raw.clip(-1, 1).hist(bins=100)
plt.title("Distribution of Raw Daily Returns")
plt.xlabel("Raw Daily Return")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(SCREENSHOT_DIR / "eda_daily_return_distribution_raw.png")
plt.close()

# 2. Cleaned distribution of daily returns
plt.figure(figsize=(10, 5))
clean_returns_cleaned.hist(bins=100)
plt.title("Distribution of Cleaned Daily Returns")
plt.xlabel("Cleaned Daily Return")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(SCREENSHOT_DIR / "eda_daily_return_distribution_clean.png")
plt.close()

# 3. Top 10 most active stocks
plt.figure(figsize=(10, 5))
plt.bar(top_active_stocks[ticker_col].astype(str), top_active_stocks["record_count"])
plt.title("Top 10 Most Active Stocks (Liquidity Proxy)")
plt.xlabel("Ticker")
plt.ylabel("Number of Records")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(SCREENSHOT_DIR / "eda_top_active_stocks.png")
plt.close()

# 4. Top 10 by cleaned average daily return
top_return_chart = ticker_summary.head(10).copy()
plt.figure(figsize=(12, 6))
plt.bar(top_return_chart[ticker_col].astype(str), top_return_chart["avg_daily_return_clean"])
plt.title("Top 10 Stocks by Average Cleaned Daily Return")
plt.xlabel("Ticker")
plt.ylabel("Average Cleaned Daily Return")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(SCREENSHOT_DIR / "eda_top_avg_daily_return.png")
plt.close()

# 5. Top 10 by cleaned volatility
vol_chart = ticker_summary.sort_values("latest_30d_volatility_clean", ascending=False).head(10).copy()
plt.figure(figsize=(12, 6))
plt.bar(vol_chart[ticker_col].astype(str), vol_chart["latest_30d_volatility_clean"])
plt.title("Top 10 Stocks by Latest 30-Day Cleaned Volatility")
plt.xlabel("Ticker")
plt.ylabel("30-Day Cleaned Volatility")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(SCREENSHOT_DIR / "eda_top_volatility.png")
plt.close()


# =========================================================
# EXPORT ENRICHED DATA
# =========================================================
df.to_csv(OUTPUT_DIR / "eda_enriched_stock_data.csv", index=False)


# =========================================================
# FINAL LOG
# =========================================================
print_section("RAW MARKET BEHAVIOUR SUMMARY")
print(raw_market_behaviour_summary)

print_section("CLEANED MARKET BEHAVIOUR SUMMARY")
print(market_behaviour_summary)

print_section("MARKET INTERPRETATION")
print(market_interpretation)

print_section("TRADING IMPLICATIONS")
print(trading_implications)

print_section("TOP ACTIVE STOCKS")
print(top_active_stocks)

print_section("EDA OUTPUTS GENERATED")
for file_name in sorted(os.listdir(OUTPUT_DIR)):
    print(f"- outputs/{file_name}")

print_section("EDA CHARTS GENERATED")
for file_name in sorted(os.listdir(SCREENSHOT_DIR)):
    if file_name.startswith("eda_") and file_name.endswith(".png"):
        print(f"- screenshots/{file_name}")

print_section("DONE")
print("EDA dashboard-ready insight engine with outlier cleaning completed successfully.")