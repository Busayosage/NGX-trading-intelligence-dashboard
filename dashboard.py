import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="NGX Trading Intelligence Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 NGX Trading Intelligence Dashboard")
st.caption("AI-powered stock analytics, trading intelligence, and quant roadmap for NGX")

# =========================================================
# PROJECT PATHS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
SCREENSHOT_DIR = BASE_DIR / "screenshots"

# =========================================================
# HELPERS
# =========================================================
def load_csv(path):
    try:
        path = Path(path)
        if path.exists():
            return pd.read_csv(path)
    except Exception as e:
        st.warning(f"Could not load {path}: {e}")
    return pd.DataFrame()


def normalize_columns(df):
    if df.empty:
        return df
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def find_col(df, candidates):
    if df.empty:
        return None

    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    for candidate in candidates:
        for col in cols:
            if candidate.lower() in col.lower():
                return col

    return None


def safe_datetime(df, col):
    if not df.empty and col and col in df.columns:
        df = df.copy()
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def safe_numeric(df, cols):
    if df.empty:
        return df
    df = df.copy()
    for col in cols:
        if col and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def min_max_scale(series):
    series = pd.to_numeric(series, errors="coerce")

    if series.isna().all():
        return pd.Series(0.5, index=series.index)

    min_val = series.min()
    max_val = series.max()

    if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
        return pd.Series(0.5, index=series.index)

    return (series - min_val) / (max_val - min_val)


def classify_risk(volatility):
    if pd.isna(volatility):
        return "Unknown"
    if volatility > 0.05:
        return "High"
    if volatility > 0.02:
        return "Moderate"
    return "Low"


def signal_to_position(signal):
    signal = str(signal).upper()
    if signal == "BUY":
        return 1
    if signal == "SELL":
        return -1
    return 0


def normalize_forecast_direction(value):
    if pd.isna(value):
        return "Unknown"

    text = str(value).strip().lower()

    if text in ["up", "upward", "bullish", "buy", "1", "1.0", "true"]:
        return "UP"
    if text in ["down", "downward", "bearish", "sell", "-1", "-1.0", "false"]:
        return "DOWN"

    try:
        num = float(text)
        if num > 0:
            return "UP"
        if num < 0:
            return "DOWN"
        return "FLAT"
    except Exception:
        return str(value)


def calculate_confidence(signal, momentum, forecast_direction):
    confidence = 50
    signal = str(signal).upper() if pd.notna(signal) else "HOLD"
    direction = normalize_forecast_direction(forecast_direction)

    if signal == "BUY":
        confidence += 20
        if pd.notna(momentum) and momentum > 0:
            confidence += 10
        if direction == "UP":
            confidence += 10

    elif signal == "SELL":
        confidence += 20
        if pd.notna(momentum) and momentum < 0:
            confidence += 10
        if direction == "DOWN":
            confidence += 10

    return min(confidence, 100)


def normalize_signal_score(signal):
    signal = str(signal).upper().strip()
    if signal == "BUY":
        return 1.0
    if signal == "HOLD":
        return 0.5
    if signal == "SELL":
        return 0.0
    return 0.5


def normalize_forecast_score(value):
    direction = normalize_forecast_direction(value)
    if direction == "UP":
        return 1.0
    if direction == "FLAT":
        return 0.5
    if direction == "DOWN":
        return 0.0
    return 0.5


def calculate_drawdown(equity_curve_series):
    running_max = equity_curve_series.cummax()
    return (equity_curve_series / running_max) - 1


def classify_strategy(momentum_threshold, confidence_threshold, holding_period, turnover):
    if momentum_threshold >= 5:
        momentum_style = "High Momentum (Aggressive)"
    elif momentum_threshold >= 3:
        momentum_style = "Moderate Momentum"
    else:
        momentum_style = "Low Momentum (Loose)"

    if confidence_threshold >= 90:
        signal_quality = "High Selectivity"
    elif confidence_threshold >= 75:
        signal_quality = "Balanced"
    else:
        signal_quality = "Loose Filtering"

    if holding_period >= 4:
        holding_style = "Medium-Term Holding"
    elif holding_period >= 2:
        holding_style = "Short-Term Swing"
    else:
        holding_style = "High-Frequency Rotation"

    if turnover >= 2:
        turnover_style = "High Turnover"
    elif turnover >= 1:
        turnover_style = "Moderate Turnover"
    else:
        turnover_style = "Low Turnover"

    return momentum_style, signal_quality, holding_style, turnover_style


# =========================================================
# RANKING + PORTFOLIO HELPERS
# =========================================================
def build_quant_ranking(latest_snapshot, momentum_col, volatility_col, forecast_col, signal_col, recent_return_col):
    if latest_snapshot.empty:
        return latest_snapshot

    ranked = latest_snapshot.copy()

    ranked["momentum_raw"] = (
        pd.to_numeric(ranked[momentum_col], errors="coerce").fillna(0.0)
        if momentum_col and momentum_col in ranked.columns else 0.0
    )

    ranked["volatility_raw"] = (
        pd.to_numeric(ranked[volatility_col], errors="coerce").fillna(0.0)
        if volatility_col and volatility_col in ranked.columns else 0.0
    )

    ranked["return_raw"] = (
        pd.to_numeric(ranked[recent_return_col], errors="coerce").fillna(0.0)
        if recent_return_col and recent_return_col in ranked.columns else 0.0
    )

    ranked["forecast_raw"] = (
        ranked[forecast_col].apply(normalize_forecast_score)
        if forecast_col and forecast_col in ranked.columns else 0.5
    )

    ranked["signal_raw"] = (
        ranked[signal_col].apply(normalize_signal_score)
        if signal_col and signal_col in ranked.columns else 0.5
    )

    ranked["momentum_score"] = min_max_scale(ranked["momentum_raw"])
    ranked["volatility_score"] = min_max_scale(ranked["volatility_raw"])
    ranked["return_score"] = min_max_scale(ranked["return_raw"])
    ranked["forecast_score"] = min_max_scale(pd.to_numeric(ranked["forecast_raw"], errors="coerce").fillna(0.5))
    ranked["signal_score"] = pd.to_numeric(ranked["signal_raw"], errors="coerce").fillna(0.5)

    ranked["ranking_score"] = (
        (0.40 * ranked["momentum_score"])
        + (0.20 * ranked["forecast_score"])
        + (0.10 * ranked["signal_score"])
        + (0.05 * ranked["return_score"])
        - (0.25 * ranked["volatility_score"])
    )

    ranked["ranking_score"] = (ranked["ranking_score"] * 100).round(2)
    return ranked.sort_values("ranking_score", ascending=False).reset_index(drop=True)


def prepare_cross_sectional_ranking(frame, momentum_col, volatility_col, forecast_col, signal_col, recent_return_col):
    if frame.empty:
        return frame

    ranked = frame.copy()

    ranked["momentum_raw"] = (
        pd.to_numeric(ranked[momentum_col], errors="coerce").fillna(0.0)
        if momentum_col and momentum_col in ranked.columns else 0.0
    )

    ranked["volatility_raw"] = (
        pd.to_numeric(ranked[volatility_col], errors="coerce").fillna(0.0)
        if volatility_col and volatility_col in ranked.columns else 0.0
    )

    ranked["return_raw"] = (
        pd.to_numeric(ranked[recent_return_col], errors="coerce").fillna(0.0)
        if recent_return_col and recent_return_col in ranked.columns else 0.0
    )

    ranked["forecast_raw"] = (
        ranked[forecast_col].apply(normalize_forecast_score)
        if forecast_col and forecast_col in ranked.columns else 0.5
    )

    ranked["signal_raw"] = (
        ranked[signal_col].apply(normalize_signal_score)
        if signal_col and signal_col in ranked.columns else 0.5
    )

    ranked["momentum_score"] = min_max_scale(ranked["momentum_raw"])
    ranked["volatility_score"] = min_max_scale(ranked["volatility_raw"])
    ranked["return_score"] = min_max_scale(ranked["return_raw"])
    ranked["forecast_score"] = min_max_scale(pd.to_numeric(ranked["forecast_raw"], errors="coerce").fillna(0.5))
    ranked["signal_score"] = pd.to_numeric(ranked["signal_raw"], errors="coerce").fillna(0.5)

    ranked["ranking_score"] = (
        (0.40 * ranked["momentum_score"])
        + (0.20 * ranked["forecast_score"])
        + (0.10 * ranked["signal_score"])
        + (0.05 * ranked["return_score"])
        - (0.25 * ranked["volatility_score"])
    )

    ranked["ranking_score"] = (ranked["ranking_score"] * 100).round(2)
    return ranked.sort_values("ranking_score", ascending=False).reset_index(drop=True)


def build_equal_weight_portfolio(
    latest_snapshot,
    top_n=5,
    ticker_col="ticker",
    close_col="close",
    signal_col=None,
    momentum_col=None,
    buy_only_mode=False,
    momentum_threshold=None,
    confidence_threshold=None
):
    if latest_snapshot.empty or "ranking_score" not in latest_snapshot.columns:
        return pd.DataFrame()

    portfolio = latest_snapshot.copy()

    if ticker_col not in portfolio.columns:
        return pd.DataFrame()

    if buy_only_mode and signal_col and signal_col in portfolio.columns:
        portfolio = portfolio[portfolio[signal_col].astype(str).str.upper() == "BUY"].copy()

    if momentum_threshold is not None and momentum_col and momentum_col in portfolio.columns:
        portfolio = portfolio[pd.to_numeric(portfolio[momentum_col], errors="coerce") > momentum_threshold].copy()

    if confidence_threshold is not None and "confidence_score" in portfolio.columns:
        portfolio = portfolio[pd.to_numeric(portfolio["confidence_score"], errors="coerce") >= confidence_threshold].copy()

    if portfolio.empty:
        return pd.DataFrame()

    portfolio = portfolio.sort_values("ranking_score", ascending=False).head(top_n).copy()
    if portfolio.empty:
        return pd.DataFrame()

    portfolio["portfolio_weight"] = 1 / len(portfolio)
    portfolio["portfolio_weight_pct"] = (portfolio["portfolio_weight"] * 100).round(2)
    portfolio["selection_reason"] = "Top ranked candidate"

    display_cols = [
        ticker_col,
        close_col,
        signal_col,
        momentum_col,
        "confidence_score",
        "risk_level",
        "ranking_score",
        "portfolio_weight_pct",
        "selection_reason"
    ]
    display_cols = [col for col in display_cols if col is not None and col in portfolio.columns]

    return portfolio[display_cols].reset_index(drop=True)


def select_with_turnover_buffer(
    ranked_day,
    previous_holdings_info,
    top_n,
    ticker_col,
    signal_col,
    momentum_col,
    replacement_buffer,
    min_holding_period=1
):
    if ranked_day.empty or ticker_col not in ranked_day.columns:
        return []

    ranked_day = ranked_day.copy().reset_index(drop=True)
    selected_universe = ranked_day.head(top_n).copy()
    if selected_universe.empty:
        return []

    cutoff_score = selected_universe["ranking_score"].iloc[-1]
    kept = []
    previous_holdings_info = previous_holdings_info or {}

    for ticker, holding_age in previous_holdings_info.items():
        row = ranked_day[ranked_day[ticker_col].astype(str) == str(ticker)]
        if row.empty:
            continue

        row = row.iloc[0]

        if signal_col and signal_col in ranked_day.columns:
            if str(row[signal_col]).upper() != "BUY":
                continue

        if momentum_col and momentum_col in ranked_day.columns:
            if pd.isna(row[momentum_col]):
                continue

        force_keep = holding_age < min_holding_period
        score_keep = row["ranking_score"] >= (cutoff_score - replacement_buffer)

        if force_keep or score_keep:
            kept.append(str(row[ticker_col]))

    kept = list(dict.fromkeys(kept))[:top_n]
    fill_candidates = ranked_day[~ranked_day[ticker_col].astype(str).isin(kept)].copy()
    fill_needed = max(top_n - len(kept), 0)
    fillers = fill_candidates.head(fill_needed)[ticker_col].astype(str).tolist()

    final_selection = kept + fillers
    return final_selection[:top_n]


def backtest_dynamic_equal_weight_portfolio(
    df,
    ticker_col,
    date_col,
    close_col,
    momentum_col,
    volatility_col,
    forecast_col,
    signal_col,
    top_n=5,
    rebalance_frequency="weekly",
    buy_only=True,
    momentum_threshold=None,
    replacement_buffer=2.0,
    confidence_threshold=None,
    transaction_cost=0.0010,
    min_holding_period=1
):
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    data = df.copy().sort_values([ticker_col, date_col])
    data["forward_return"] = data.groupby(ticker_col)[close_col].pct_change().shift(-1)

    if momentum_col and momentum_col in data.columns:
        data[momentum_col] = data.groupby(ticker_col)[momentum_col].shift(2)
    if volatility_col and volatility_col in data.columns:
        data[volatility_col] = data.groupby(ticker_col)[volatility_col].shift(2)
    if forecast_col and forecast_col in data.columns:
        data[forecast_col] = data.groupby(ticker_col)[forecast_col].shift(2)
    if signal_col and signal_col in data.columns:
        data[signal_col] = data.groupby(ticker_col)[signal_col].shift(2)

    recent_return_col = "comparison_daily_return_clean" if "comparison_daily_return_clean" in data.columns else None
    if recent_return_col and recent_return_col in data.columns:
        data[recent_return_col] = data.groupby(ticker_col)[recent_return_col].shift(2)

    data = data.copy()
    data["rebalance_date"] = data[date_col]

    if rebalance_frequency == "weekly":
        iso_calendar = data[date_col].dt.isocalendar()
        data["rebalance_group"] = iso_calendar["year"].astype(str) + "-W" + iso_calendar["week"].astype(str).str.zfill(2)
    else:
        data["rebalance_group"] = data[date_col].astype(str)

    portfolio_rows = []
    previous_holdings_info = {}

    for _, group_df in data.groupby("rebalance_group"):
        group_df = group_df.copy().sort_values(date_col)
        day_slice = group_df.copy()

        required_rank_inputs = []
        if momentum_col and momentum_col in day_slice.columns:
            required_rank_inputs.append(momentum_col)
        if volatility_col and volatility_col in day_slice.columns:
            required_rank_inputs.append(volatility_col)
        if forecast_col and forecast_col in day_slice.columns:
            required_rank_inputs.append(forecast_col)
        if signal_col and signal_col in day_slice.columns:
            required_rank_inputs.append(signal_col)

        if required_rank_inputs:
            day_slice = day_slice.dropna(subset=required_rank_inputs, how="all")

        if day_slice.empty:
            previous_holdings_info = {ticker: age + 1 for ticker, age in previous_holdings_info.items()}
            continue

        day_slice = (
            day_slice.sort_values([ticker_col, date_col])
            .groupby(ticker_col, as_index=False)
            .head(1)
            .copy()
        )

        day_slice["confidence_score"] = day_slice.apply(
            lambda row: calculate_confidence(
                row[signal_col] if signal_col and signal_col in day_slice.columns else "HOLD",
                row[momentum_col] if momentum_col and momentum_col in day_slice.columns else np.nan,
                row[forecast_col] if forecast_col and forecast_col in day_slice.columns else np.nan
            ),
            axis=1
        )

        if buy_only and signal_col and signal_col in day_slice.columns:
            day_slice = day_slice[day_slice[signal_col].astype(str).str.upper() == "BUY"].copy()

        if momentum_threshold is not None and momentum_col and momentum_col in day_slice.columns:
            day_slice = day_slice[pd.to_numeric(day_slice[momentum_col], errors="coerce") > momentum_threshold].copy()

        if confidence_threshold is not None and "confidence_score" in day_slice.columns:
            day_slice = day_slice[pd.to_numeric(day_slice["confidence_score"], errors="coerce") >= confidence_threshold].copy()

        if day_slice.empty:
            previous_holdings_info = {ticker: age + 1 for ticker, age in previous_holdings_info.items()}
            continue

        ranked_day = prepare_cross_sectional_ranking(
            frame=day_slice,
            momentum_col=momentum_col,
            volatility_col=volatility_col,
            forecast_col=forecast_col,
            signal_col=signal_col,
            recent_return_col=recent_return_col
        )

        if ranked_day.empty:
            previous_holdings_info = {ticker: age + 1 for ticker, age in previous_holdings_info.items()}
            continue

        if "confidence_score" not in ranked_day.columns and "confidence_score" in day_slice.columns:
            ranked_day = ranked_day.merge(
                day_slice[[ticker_col, "confidence_score"]],
                on=ticker_col,
                how="left"
            )

        selected_tickers = select_with_turnover_buffer(
            ranked_day=ranked_day,
            previous_holdings_info=previous_holdings_info,
            top_n=top_n,
            ticker_col=ticker_col,
            signal_col=signal_col,
            momentum_col=momentum_col,
            replacement_buffer=replacement_buffer,
            min_holding_period=min_holding_period
        )

        if not selected_tickers:
            previous_holdings_info = {ticker: age + 1 for ticker, age in previous_holdings_info.items()}
            continue

        selected = ranked_day[ranked_day[ticker_col].astype(str).isin(selected_tickers)].copy()
        if selected.empty:
            previous_holdings_info = {ticker: age + 1 for ticker, age in previous_holdings_info.items()}
            continue

        selected["selection_order"] = pd.Categorical(
            selected[ticker_col].astype(str),
            categories=selected_tickers,
            ordered=True
        )
        selected = selected.sort_values("selection_order").drop(columns=["selection_order"])

        selected = selected.dropna(subset=["forward_return"]).copy()
        if selected.empty:
            previous_holdings_info = {ticker: age + 1 for ticker, age in previous_holdings_info.items()}
            continue

        selected["forward_return"] = selected["forward_return"].clip(-0.15, 0.15)
        selected["portfolio_weight"] = 1 / len(selected)
        selected["gross_weighted_forward_return"] = selected["portfolio_weight"] * selected["forward_return"]
        selected["weighted_forward_return"] = selected["gross_weighted_forward_return"] - (selected["portfolio_weight"] * transaction_cost)
        selected["rebalance_date"] = selected[date_col].min()

        keep_cols = [
            ticker_col,
            date_col,
            "rebalance_date",
            signal_col,
            momentum_col,
            "confidence_score",
            "ranking_score",
            "portfolio_weight",
            "forward_return",
            "gross_weighted_forward_return",
            "weighted_forward_return"
        ]
        keep_cols = [c for c in keep_cols if c in selected.columns]

        portfolio_rows.append(selected[keep_cols])

        new_holdings_info = {}
        for ticker in selected[ticker_col].astype(str).tolist():
            if ticker in previous_holdings_info:
                new_holdings_info[ticker] = previous_holdings_info[ticker] + 1
            else:
                new_holdings_info[ticker] = 1
        previous_holdings_info = new_holdings_info

    if not portfolio_rows:
        return pd.DataFrame(), pd.DataFrame()

    portfolio_holdings = pd.concat(portfolio_rows, ignore_index=True)

    portfolio_returns = (
        portfolio_holdings.groupby("rebalance_date", as_index=False)
        .agg(
            portfolio_return=("weighted_forward_return", "sum"),
            gross_portfolio_return=("gross_weighted_forward_return", "sum")
        )
        .sort_values("rebalance_date")
    )

    portfolio_returns["equity_curve"] = (1 + portfolio_returns["portfolio_return"].fillna(0)).cumprod()
    portfolio_returns["cumulative_return"] = portfolio_returns["equity_curve"] - 1

    return portfolio_returns, portfolio_holdings


def calculate_portfolio_diagnostics(portfolio_returns, portfolio_holdings, ticker_col):
    diagnostics = {}

    if portfolio_returns.empty:
        return diagnostics, pd.DataFrame(), portfolio_returns

    returns = portfolio_returns["portfolio_return"].fillna(0)

    diagnostics["net_total_return_pct"] = (portfolio_returns["equity_curve"].iloc[-1] - 1) * 100
    diagnostics["avg_period_return"] = returns.mean()
    diagnostics["volatility"] = returns.std()
    diagnostics["win_rate_pct"] = (returns > 0).mean() * 100
    diagnostics["best_period_return"] = returns.max()
    diagnostics["worst_period_return"] = returns.min()

    if pd.notna(diagnostics["volatility"]) and diagnostics["volatility"] != 0:
        diagnostics["sharpe_ratio"] = returns.mean() / returns.std()
    else:
        diagnostics["sharpe_ratio"] = np.nan

    portfolio_returns = portfolio_returns.copy()
    portfolio_returns["drawdown"] = calculate_drawdown(portfolio_returns["equity_curve"])
    diagnostics["max_drawdown_pct"] = portfolio_returns["drawdown"].min() * 100

    diagnostics["gross_total_return_pct"] = np.nan
    if "gross_portfolio_return" in portfolio_returns.columns:
        portfolio_returns["gross_equity_curve"] = (1 + portfolio_returns["gross_portfolio_return"].fillna(0)).cumprod()
        diagnostics["gross_total_return_pct"] = (portfolio_returns["gross_equity_curve"].iloc[-1] - 1) * 100

    turnover_df = pd.DataFrame()

    if not portfolio_holdings.empty and "rebalance_date" in portfolio_holdings.columns and ticker_col in portfolio_holdings.columns:
        holdings_by_date = (
            portfolio_holdings.groupby("rebalance_date")[ticker_col]
            .apply(lambda s: set(s.astype(str)))
            .reset_index(name="holdings_set")
            .sort_values("rebalance_date")
            .reset_index(drop=True)
        )

        turnover_records = []
        prev_set = None

        for _, row in holdings_by_date.iterrows():
            current_set = row["holdings_set"]

            if prev_set is None:
                names_in = len(current_set)
                names_out = 0
                turnover_rate = 1.0 if len(current_set) > 0 else 0.0
            else:
                names_in = len(current_set - prev_set)
                names_out = len(prev_set - current_set)
                base_n = max(len(current_set), 1)
                turnover_rate = (names_in + names_out) / base_n

            turnover_records.append({
                "rebalance_date": row["rebalance_date"],
                "names_added": names_in,
                "names_removed": names_out,
                "turnover_rate": turnover_rate
            })

            prev_set = current_set

        turnover_df = pd.DataFrame(turnover_records)

        if not turnover_df.empty:
            diagnostics["avg_turnover_rate"] = turnover_df["turnover_rate"].mean()
            diagnostics["max_turnover_rate"] = turnover_df["turnover_rate"].max()
        else:
            diagnostics["avg_turnover_rate"] = np.nan
            diagnostics["max_turnover_rate"] = np.nan
    else:
        diagnostics["avg_turnover_rate"] = np.nan
        diagnostics["max_turnover_rate"] = np.nan

    return diagnostics, turnover_df, portfolio_returns


def run_parameter_optimization(
    df,
    ticker_col,
    date_col,
    close_col,
    momentum_col,
    volatility_col,
    forecast_col,
    signal_col,
    replacement_buffer,
    confidence_threshold,
    transaction_cost,
    momentum_values=None,
    holding_values=None,
    top_n_values=None,
    rebalance_frequency="weekly",
    buy_only=True
):
    if momentum_values is None:
        momentum_values = [2.0, 3.0, 4.0]

    if holding_values is None:
        holding_values = [1, 2, 3]

    if top_n_values is None:
        top_n_values = [3, 5]

    results = []

    for momentum_threshold_test in momentum_values:
        for min_holding_period_test in holding_values:
            for top_n_test in top_n_values:
                test_returns, test_holdings = backtest_dynamic_equal_weight_portfolio(
                    df=df,
                    ticker_col=ticker_col,
                    date_col=date_col,
                    close_col=close_col,
                    momentum_col=momentum_col,
                    volatility_col=volatility_col,
                    forecast_col=forecast_col,
                    signal_col=signal_col,
                    top_n=top_n_test,
                    rebalance_frequency=rebalance_frequency,
                    buy_only=buy_only,
                    momentum_threshold=momentum_threshold_test,
                    replacement_buffer=replacement_buffer,
                    confidence_threshold=confidence_threshold,
                    transaction_cost=transaction_cost,
                    min_holding_period=min_holding_period_test
                )

                if test_returns.empty:
                    continue

                diagnostics, _, _ = calculate_portfolio_diagnostics(
                    portfolio_returns=test_returns,
                    portfolio_holdings=test_holdings,
                    ticker_col=ticker_col
                )

                results.append({
                    "momentum_threshold": momentum_threshold_test,
                    "min_holding_period": min_holding_period_test,
                    "top_n": top_n_test,
                    "net_total_return_pct": diagnostics.get("net_total_return_pct", np.nan),
                    "gross_total_return_pct": diagnostics.get("gross_total_return_pct", np.nan),
                    "sharpe_ratio": diagnostics.get("sharpe_ratio", np.nan),
                    "max_drawdown_pct": diagnostics.get("max_drawdown_pct", np.nan),
                    "win_rate_pct": diagnostics.get("win_rate_pct", np.nan),
                    "avg_turnover_rate": diagnostics.get("avg_turnover_rate", np.nan),
                })

    if not results:
        return pd.DataFrame()

    optimization_df = pd.DataFrame(results).sort_values(
        ["net_total_return_pct", "sharpe_ratio"],
        ascending=[False, False]
    ).reset_index(drop=True)

    return optimization_df


def latest_per_ticker(df, ticker_col, date_col):
    if df.empty or ticker_col is None or date_col is None:
        return pd.DataFrame()

    return (
        df.sort_values([ticker_col, date_col])
        .groupby(ticker_col, as_index=False)
        .tail(1)
        .copy()
    )


def filter_by_ticker(df, ticker_col, ticker_value):
    if df.empty or ticker_col is None:
        return pd.DataFrame()
    return df[df[ticker_col].astype(str) == str(ticker_value)].copy()


def explain_stock(stock_code, latest_df):
    if latest_df.empty:
        return "No latest stock data is available."

    ticker_col_local = find_col(latest_df, ["ticker", "symbol", "stock", "security", "ticker_clean"])
    signal_col_local = find_col(latest_df, ["signal_combined", "signal", "trade_signal", "action"])
    close_col_local = find_col(latest_df, ["close", "closing_price", "adj_close", "price"])
    momentum_col_local = find_col(latest_df, ["momentum"])
    volatility_col_local = find_col(latest_df, ["volatility", "vol"])
    forecast_col_local = find_col(latest_df, ["forecast_direction", "prediction_direction", "prediction", "forecast", "pred_direction_prev_close"])

    if ticker_col_local is None:
        return "Ticker column could not be found."

    row_df = latest_df[latest_df[ticker_col_local].astype(str).str.upper() == stock_code.upper()]
    if row_df.empty:
        return f"I could not find data for {stock_code.upper()}."

    row = row_df.iloc[0]

    signal_value = row[signal_col_local] if signal_col_local else "HOLD"
    close_value = row[close_col_local] if close_col_local else np.nan
    momentum_value = row[momentum_col_local] if momentum_col_local else np.nan
    volatility_value = row[volatility_col_local] if volatility_col_local else np.nan
    forecast_value = normalize_forecast_direction(row[forecast_col_local]) if forecast_col_local else "Unknown"
    confidence_value = row["confidence_score"] if "confidence_score" in row.index else np.nan
    risk_value = row["risk_level"] if "risk_level" in row.index else "Unknown"

    lines = [f"Signal is **{signal_value}**."]

    if pd.notna(close_value):
        lines.append(f"Latest price is **{close_value:,.2f}**.")

    if pd.notna(confidence_value):
        lines.append(f"Confidence is **{confidence_value:.0f}/100**.")

    lines.append(f"Risk is **{risk_value}**.")

    reason_parts = []
    if pd.notna(momentum_value):
        reason_parts.append(f"momentum is **{momentum_value:.4f}**")
    if pd.notna(volatility_value):
        reason_parts.append(f"volatility is **{volatility_value:.4f}**")
    if forecast_value != "Unknown":
        reason_parts.append(f"forecast direction is **{forecast_value}**")

    if reason_parts:
        lines.append("Reasoning: " + ", ".join(reason_parts) + ".")

    return " ".join(lines)


def compare_stocks_text(stock_a, stock_b, latest_df, summary_df):
    ticker_col_local = find_col(latest_df, ["ticker", "symbol", "stock", "security", "ticker_clean"])
    signal_col_local = find_col(latest_df, ["signal_combined", "signal", "trade_signal", "action"])

    if latest_df.empty or ticker_col_local is None:
        return f"I could not compare {stock_a} and {stock_b}."

    a_latest = latest_df[latest_df[ticker_col_local].astype(str).str.upper() == stock_a.upper()]
    b_latest = latest_df[latest_df[ticker_col_local].astype(str).str.upper() == stock_b.upper()]

    if a_latest.empty or b_latest.empty:
        return f"I could not compare {stock_a.upper()} and {stock_b.upper()} because one or both are missing."

    a_row = a_latest.iloc[0]
    b_row = b_latest.iloc[0]

    a_summary = summary_df[summary_df["ticker"].astype(str).str.upper() == stock_a.upper()] if not summary_df.empty else pd.DataFrame()
    b_summary = summary_df[summary_df["ticker"].astype(str).str.upper() == stock_b.upper()] if not summary_df.empty else pd.DataFrame()

    a_avg = a_summary["avg_return"].iloc[0] if not a_summary.empty else np.nan
    b_avg = b_summary["avg_return"].iloc[0] if not b_summary.empty else np.nan

    a_conf = a_row["confidence_score"] if "confidence_score" in a_row.index else np.nan
    b_conf = b_row["confidence_score"] if "confidence_score" in b_row.index else np.nan

    a_sig = a_row[signal_col_local] if signal_col_local else "N/A"
    b_sig = b_row[signal_col_local] if signal_col_local else "N/A"

    a_avg_text = f"{a_avg:.4f}" if pd.notna(a_avg) else "N/A"
    b_avg_text = f"{b_avg:.4f}" if pd.notna(b_avg) else "N/A"
    a_conf_text = f"{a_conf:.0f}/100" if pd.notna(a_conf) else "N/A"
    b_conf_text = f"{b_conf:.0f}/100" if pd.notna(b_conf) else "N/A"

    return (
        f"Comparison:\n"
        f"- {stock_a.upper()}: signal {a_sig}, confidence {a_conf_text}, avg return {a_avg_text}\n"
        f"- {stock_b.upper()}: signal {b_sig}, confidence {b_conf_text}, avg return {b_avg_text}"
    )


def answer_ai_query(query, latest_df, summary_df, eda_interpretation_df=None, eda_trading_df=None, active_df=None):
    q = query.strip().lower()

    ticker_col_local = find_col(latest_df, ["ticker", "symbol", "stock", "security", "ticker_clean"])
    signal_col_local = find_col(latest_df, ["signal_combined", "signal", "trade_signal", "action"])
    volatility_col_local = find_col(latest_df, ["volatility", "vol"])

    if latest_df.empty and summary_df.empty:
        return "I do not have enough market data loaded yet."

    if "top buy" in q or "best buy" in q or "buy opportunities" in q:
        if latest_df.empty or signal_col_local is None or ticker_col_local is None:
            return "I could not find current BUY opportunities."
        buys = latest_df[latest_df[signal_col_local].astype(str).str.upper() == "BUY"].copy()
        if buys.empty:
            return "There are no current BUY opportunities."
        buys = buys.sort_values("confidence_score", ascending=False) if "confidence_score" in buys.columns else buys
        return "Top BUY stocks: " + ", ".join(buys[ticker_col_local].astype(str).head(5).tolist())

    if "best performing" in q or "top performing" in q or "leaderboard" in q:
        if summary_df.empty:
            return "Strategy summary is not available."
        return "Best performing stocks: " + ", ".join(summary_df["ticker"].astype(str).head(5).tolist())

    if "risk" in q or "riskiest" in q:
        if latest_df.empty or volatility_col_local is None or ticker_col_local is None:
            return "Risk data is not available."
        riskiest = latest_df.sort_values(volatility_col_local, ascending=False).iloc[0]
        return f"Riskiest stock is {riskiest[ticker_col_local]} with volatility {riskiest[volatility_col_local]:.4f}."

    if "active stock" in q or "most active" in q or "liquidity" in q:
        if active_df is not None and not active_df.empty:
            col0 = active_df.columns[0]
            return "Most active stocks: " + ", ".join(active_df[col0].astype(str).head(5).tolist())
        return "Active stock data is not available."

    if "market regime" in q or "what kind of market" in q or "market insight" in q:
        if eda_interpretation_df is not None and not eda_interpretation_df.empty:
            if "insight" in eda_interpretation_df.columns:
                return " | ".join(eda_interpretation_df["insight"].astype(str).head(3).tolist())
            return " | ".join(eda_interpretation_df.iloc[:, 0].astype(str).head(3).tolist())
        return "EDA market interpretation is not available yet."

    if "strategy should i use" in q or "trading implication" in q or "what strategy" in q:
        if eda_trading_df is not None and not eda_trading_df.empty and "category" in eda_trading_df.columns and "strategy" in eda_trading_df.columns:
            favors = eda_trading_df[eda_trading_df["category"] == "favors"]["strategy"].astype(str).tolist()
            avoids = eda_trading_df[eda_trading_df["category"] == "avoid"]["strategy"].astype(str).tolist()
            return f"Favours: {', '.join(favors[:4])}. Avoid: {', '.join(avoids[:3])}."
        return "Trading implication data is not available."

    if "compare" in q:
        tickers_found = re.findall(r"\b[A-Z]{2,20}\b", query.upper())
        if len(tickers_found) >= 2:
            return compare_stocks_text(tickers_found[0], tickers_found[1], latest_df, summary_df)
        return "Please provide two tickers to compare, for example: Compare GTCO and DANGSUGAR."

    if "why is" in q or "explain" in q:
        tickers_found = re.findall(r"\b[A-Z]{2,20}\b", query.upper())
        if len(tickers_found) >= 1:
            return explain_stock(tickers_found[0], latest_df)
        return "Please include a ticker, for example: Why is GTCO a BUY?"

    return "Ask about BUY opportunities, best performers, market regime, active stocks, risk, or compare two stocks."


# =========================================================
# LOAD MAIN DATA
# =========================================================
raw_df = normalize_columns(load_csv(DATA_DIR / "stock_data.csv"))
signal_df = normalize_columns(load_csv(OUTPUT_DIR / "signal_output.csv"))
tech_df = normalize_columns(load_csv(OUTPUT_DIR / "technical_indicators.csv"))

df = signal_df.copy() if not signal_df.empty else raw_df.copy()

if df.empty:
    st.error("No data found. Please make sure your CSV files exist in data/ or outputs/.")
    st.stop()

ticker_col = find_col(df, ["ticker_clean", "ticker", "symbol", "stock", "security"])
date_col = find_col(df, ["date", "datetime"])
close_col = find_col(df, ["close", "closing_price", "adj_close", "price"])
signal_col = find_col(df, ["signal_combined", "signal", "trade_signal", "action"])
return_col = find_col(df, ["return", "daily_return", "pct_return"])
momentum_col = find_col(df, ["momentum"])
volatility_col = find_col(df, ["volatility", "vol"])
forecast_col = find_col(df, ["forecast_direction", "prediction_direction", "prediction", "forecast", "pred_direction_prev_close"])

if ticker_col is None or date_col is None:
    st.error("Ticker or date column is missing from your data.")
    st.stop()

df = safe_datetime(df, date_col)
df = safe_numeric(df, [close_col, return_col, momentum_col, volatility_col, forecast_col])

if not tech_df.empty:
    tech_df = normalize_columns(tech_df)
    tech_ticker_col = find_col(tech_df, ["ticker_clean", "ticker", "symbol", "stock", "security"])
    tech_date_col = find_col(tech_df, ["date", "datetime"])
    tech_momentum_col = find_col(tech_df, ["momentum"])
    tech_volatility_col = find_col(tech_df, ["volatility", "vol"])

    tech_df = safe_datetime(tech_df, tech_date_col)
    tech_df = safe_numeric(tech_df, [tech_momentum_col, tech_volatility_col])

    if all([tech_ticker_col, tech_date_col]):
        tech_keep = [tech_ticker_col, tech_date_col]

        if momentum_col is None and tech_momentum_col:
            tech_keep.append(tech_momentum_col)
        if volatility_col is None and tech_volatility_col:
            tech_keep.append(tech_volatility_col)

        tech_subset = tech_df[tech_keep].copy().rename(
            columns={tech_ticker_col: ticker_col, tech_date_col: date_col}
        )
        df = df.merge(tech_subset, on=[ticker_col, date_col], how="left")

        momentum_col = find_col(df, ["momentum"])
        volatility_col = find_col(df, ["volatility", "vol"])

if return_col is None:
    if close_col is None:
        st.error("Neither return column nor close price column was found.")
        st.stop()
    df["return_calc"] = df.groupby(ticker_col)[close_col].pct_change()
    return_col = "return_calc"

if signal_col is None:
    if momentum_col is not None:
        df["signal"] = np.where(
            df[momentum_col] > 0.02,
            "BUY",
            np.where(df[momentum_col] < -0.02, "SELL", "HOLD")
        )
    else:
        df["signal"] = "HOLD"
    signal_col = "signal"

df = df.sort_values([ticker_col, date_col]).copy()

# =========================================================
# COMPARISON HELPER RETURNS (FIXED - NO LOOKAHEAD)
# =========================================================
df["comparison_daily_return"] = df.groupby(ticker_col)[close_col].pct_change().shift(1)
df["comparison_daily_return_clean"] = df["comparison_daily_return"].clip(-0.20, 0.20)

# =========================================================
# BACKTEST BASELINE
# =========================================================
df["position"] = df[signal_col].apply(signal_to_position)
df["position_shifted"] = df.groupby(ticker_col)["position"].shift(1).fillna(0)
df["strategy_return"] = df["position_shifted"] * df[return_col].fillna(0)

summary = (
    df.groupby(ticker_col, as_index=False)
    .agg(
        total_return=("strategy_return", "sum"),
        avg_return=("strategy_return", "mean"),
        trade_count=("strategy_return", "count")
    )
    .rename(columns={ticker_col: "ticker"})
    .sort_values("avg_return", ascending=False)
)

# =========================================================
# LATEST SNAPSHOT
# =========================================================
latest_snapshot = latest_per_ticker(df, ticker_col, date_col)

if not latest_snapshot.empty:
    latest_snapshot["confidence_score"] = latest_snapshot.apply(
        lambda row: calculate_confidence(
            row[signal_col] if signal_col else "HOLD",
            row[momentum_col] if momentum_col and momentum_col in latest_snapshot.columns else np.nan,
            row[forecast_col] if forecast_col and forecast_col in latest_snapshot.columns else np.nan
        ),
        axis=1
    )

    if volatility_col and volatility_col in latest_snapshot.columns:
        latest_snapshot["risk_level"] = latest_snapshot[volatility_col].apply(classify_risk)
    else:
        latest_snapshot["risk_level"] = "Unknown"

    recent_return_col = "comparison_daily_return_clean" if "comparison_daily_return_clean" in latest_snapshot.columns else return_col

    latest_snapshot = build_quant_ranking(
        latest_snapshot=latest_snapshot,
        momentum_col=momentum_col,
        volatility_col=volatility_col,
        forecast_col=forecast_col,
        signal_col=signal_col,
        recent_return_col=recent_return_col
    )

# =========================================================
# LOAD OPTIONAL EDA OUTPUTS
# =========================================================
eda_market_summary_lines = normalize_columns(load_csv(OUTPUT_DIR / "eda_market_summary_lines.csv"))
eda_interpretation_lines = normalize_columns(load_csv(OUTPUT_DIR / "eda_interpretation_lines.csv"))
eda_market_behaviour = normalize_columns(load_csv(OUTPUT_DIR / "eda_market_behaviour_summary.csv"))
eda_market_interpretation = normalize_columns(load_csv(OUTPUT_DIR / "eda_market_interpretation.csv"))
eda_trading_implications = normalize_columns(load_csv(OUTPUT_DIR / "eda_trading_implications.csv"))
eda_top_active_stocks = normalize_columns(load_csv(OUTPUT_DIR / "eda_top_active_stocks.csv"))
eda_top_gainers = normalize_columns(load_csv(OUTPUT_DIR / "eda_top_gainers.csv"))
eda_top_losers = normalize_columns(load_csv(OUTPUT_DIR / "eda_top_losers.csv"))

# =========================================================
# SIDEBAR CONTROLS
# =========================================================
st.sidebar.header("Controls")

available_tickers = []
if not latest_snapshot.empty and ticker_col in latest_snapshot.columns:
    available_tickers = sorted(latest_snapshot[ticker_col].dropna().astype(str).unique().tolist())

if not available_tickers:
    available_tickers = sorted(df[ticker_col].dropna().astype(str).unique().tolist())

selected_ticker = st.sidebar.selectbox(
    "Select Stock",
    options=available_tickers if available_tickers else ["No stocks available"]
)

momentum_threshold = st.sidebar.slider(
    "Momentum Threshold",
    min_value=0.0,
    max_value=10.0,
    value=3.0,
    step=0.5
)

dynamic_top_n = st.sidebar.slider(
    "Backtest Portfolio Size (Top N)",
    min_value=3,
    max_value=10,
    value=3,
    step=1
)

replacement_buffer = st.sidebar.slider(
    "Replacement Buffer",
    min_value=0.0,
    max_value=15.0,
    value=2.0,
    step=0.5
)

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=50,
    max_value=100,
    value=90,
    step=5
)

transaction_cost = st.sidebar.slider(
    "Transaction Cost",
    min_value=0.0010,
    max_value=0.0030,
    value=0.0010,
    step=0.0005,
    format="%.4f"
)

min_holding_period = st.sidebar.slider(
    "Minimum Holding Period (weeks)",
    min_value=1,
    max_value=8,
    value=1,
    step=1
)

default_compare = available_tickers[:3] if len(available_tickers) >= 3 else available_tickers

selected_stock_df = filter_by_ticker(df, ticker_col, selected_ticker)
selected_latest_df = filter_by_ticker(latest_snapshot, ticker_col, selected_ticker)

# =========================================================
# DYNAMIC BACKTEST
# =========================================================
rebalance_frequency = "weekly"
buy_only_mode = True

dynamic_portfolio_returns, dynamic_portfolio_holdings = backtest_dynamic_equal_weight_portfolio(
    df=df,
    ticker_col=ticker_col,
    date_col=date_col,
    close_col=close_col,
    momentum_col=momentum_col,
    volatility_col=volatility_col,
    forecast_col=forecast_col,
    signal_col=signal_col,
    top_n=dynamic_top_n,
    rebalance_frequency=rebalance_frequency,
    buy_only=buy_only_mode,
    momentum_threshold=momentum_threshold,
    replacement_buffer=replacement_buffer,
    confidence_threshold=confidence_threshold,
    transaction_cost=transaction_cost,
    min_holding_period=min_holding_period
)

portfolio_diagnostics = {}
turnover_df = pd.DataFrame()
portfolio_returns_with_diag = dynamic_portfolio_returns.copy() if dynamic_portfolio_returns is not None else pd.DataFrame()

if dynamic_portfolio_returns is not None and not dynamic_portfolio_returns.empty:
    portfolio_diagnostics, turnover_df, portfolio_returns_with_diag = calculate_portfolio_diagnostics(
        portfolio_returns=dynamic_portfolio_returns,
        portfolio_holdings=dynamic_portfolio_holdings,
        ticker_col=ticker_col
    )

# =========================================================
# KPI HELPERS
# =========================================================
def get_latest_price(df_local):
    if df_local.empty or close_col is None:
        return np.nan
    return df_local.sort_values(date_col)[close_col].iloc[-1]


def get_latest_signal(df_local):
    if df_local.empty or signal_col is None:
        return "N/A"
    return str(df_local.sort_values(date_col)[signal_col].iloc[-1])


def get_latest_confidence(df_local):
    if df_local.empty or "confidence_score" not in df_local.columns:
        return np.nan
    return df_local["confidence_score"].iloc[-1]


def get_latest_risk(df_local):
    if df_local.empty or "risk_level" not in df_local.columns:
        return "Unknown"
    return df_local["risk_level"].iloc[-1]


# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Market", "Portfolio", "AI Chat", "Quant Control Center"]
)

# =========================================================
# OVERVIEW
# =========================================================
with tab1:
    st.subheader("Overview")

    c1, c2, c3, c4 = st.columns(4)

    latest_price = get_latest_price(selected_stock_df)
    latest_signal = get_latest_signal(selected_stock_df)
    latest_confidence = get_latest_confidence(selected_latest_df)
    latest_risk = get_latest_risk(selected_latest_df)

    c1.metric("Price", f"{latest_price:,.2f}" if pd.notna(latest_price) else "N/A")
    c2.metric("Signal", latest_signal)
    c3.metric("Confidence", f"{latest_confidence:.0f}/100" if pd.notna(latest_confidence) else "N/A")
    c4.metric("Risk", latest_risk)

    st.markdown("### AI Insight")
    if not selected_latest_df.empty:
        st.markdown(explain_stock(selected_ticker, latest_snapshot))
    else:
        st.info("No AI insight is available for the selected stock.")

    st.markdown("### Price Trend")
    if not selected_stock_df.empty and close_col:
        fig_price = px.line(
            selected_stock_df.sort_values(date_col),
            x=date_col,
            y=close_col,
            title=f"{selected_ticker} Price Trend"
        )
        st.plotly_chart(fig_price, use_container_width=True)
    else:
        st.info("Price trend is unavailable.")

    st.markdown("### Signals")
    if not selected_stock_df.empty and close_col and signal_col:
        fig_signal = px.scatter(
            selected_stock_df.sort_values(date_col),
            x=date_col,
            y=close_col,
            color=signal_col,
            title=f"{selected_ticker} Trading Signals"
        )
        st.plotly_chart(fig_signal, use_container_width=True)
    else:
        st.info("Signal chart is unavailable.")

# =========================================================
# MARKET
# =========================================================
with tab2:
    st.subheader("Market Intelligence")

    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown("### Top Stocks")
        top_10 = summary.head(10)

        if not top_10.empty:
            fig_top = px.bar(
                top_10,
                x="ticker",
                y="avg_return",
                title="Top 10 Stocks by Average Strategy Return"
            )
            st.plotly_chart(fig_top, use_container_width=True)
            st.dataframe(
                top_10[["ticker", "avg_return", "total_return", "trade_count"]],
                use_container_width=True
            )
        else:
            st.info("No summary data available.")

    with right_col:
        st.markdown("### Best BUY")
        if not latest_snapshot.empty and signal_col:
            buys = latest_snapshot[latest_snapshot[signal_col].astype(str).str.upper() == "BUY"].copy()

            display_cols = []
            if ticker_col in buys.columns:
                display_cols.append(ticker_col)
            if close_col and close_col in buys.columns:
                display_cols.append(close_col)
            if momentum_col and momentum_col in buys.columns:
                display_cols.append(momentum_col)
            if volatility_col and volatility_col in buys.columns:
                display_cols.append(volatility_col)
            if "confidence_score" in buys.columns:
                display_cols.append("confidence_score")
            if "risk_level" in buys.columns:
                display_cols.append("risk_level")
            if "ranking_score" in buys.columns:
                display_cols.append("ranking_score")

            buys = buys.sort_values("confidence_score", ascending=False) if "confidence_score" in buys.columns else buys

            if not buys.empty:
                st.dataframe(buys[display_cols].head(10), use_container_width=True)
            else:
                st.info("No current BUY opportunities.")
        else:
            st.info("BUY opportunity data is unavailable.")

    st.markdown("### 🧮 Quant Ranking Preview")
    if not latest_snapshot.empty:
        ranking_display_cols = [
            c for c in [
                ticker_col,
                close_col,
                signal_col,
                momentum_col,
                volatility_col,
                "confidence_score",
                "risk_level",
                "ranking_score"
            ] if c is not None and c in latest_snapshot.columns
        ]
        st.dataframe(latest_snapshot[ranking_display_cols].head(10), use_container_width=True)
    else:
        st.info("Ranking preview is not available.")

    st.markdown("### 📊 Multi-Stock Comparison")
    st.caption("Compare selected stocks using a chosen time window, normalized price performance, daily return distribution, and summary metrics.")

    compare_control_col1, compare_control_col2 = st.columns([2, 1])

    with compare_control_col1:
        market_compare_selection = st.multiselect(
            "Choose stocks to compare",
            options=available_tickers,
            default=default_compare,
            key="market_compare_selection"
        )

    with compare_control_col2:
        comparison_period = st.selectbox(
            "Comparison period",
            options=["3M", "6M", "1Y", "3Y", "Max"],
            index=2,
            key="comparison_period"
        )

    compare_df = df[df[ticker_col].astype(str).isin([str(t) for t in market_compare_selection])].copy()

    if len(market_compare_selection) < 2:
        st.warning("Please select at least 2 stocks to compare.")
    elif compare_df.empty:
        st.info("No comparison data available for selected stocks.")
    else:
        compare_df = compare_df.sort_values([ticker_col, date_col]).copy()
        latest_compare_date = compare_df[date_col].max()

        if pd.notna(latest_compare_date) and comparison_period != "Max":
            if comparison_period == "3M":
                start_date = latest_compare_date - pd.DateOffset(months=3)
            elif comparison_period == "6M":
                start_date = latest_compare_date - pd.DateOffset(months=6)
            elif comparison_period == "1Y":
                start_date = latest_compare_date - pd.DateOffset(years=1)
            elif comparison_period == "3Y":
                start_date = latest_compare_date - pd.DateOffset(years=3)
            else:
                start_date = None

            if start_date is not None:
                compare_df = compare_df[compare_df[date_col] >= start_date].copy()

        compare_df = compare_df.sort_values([ticker_col, date_col]).copy()

        st.markdown("#### 📈 Normalized Price Performance")
        norm_df = compare_df.copy()
        norm_df["normalized_price"] = norm_df.groupby(ticker_col)[close_col].transform(
            lambda s: (s / s.iloc[0] * 100) if len(s) > 0 and pd.notna(s.iloc[0]) and s.iloc[0] != 0 else np.nan
        )

        fig_norm = px.line(
            norm_df,
            x=date_col,
            y="normalized_price",
            color=ticker_col,
            title=f"Growth of 100-Base Investment ({comparison_period})"
        )
        st.plotly_chart(fig_norm, use_container_width=True)

        st.markdown("#### 📉 Daily Returns Comparison")
        returns_plot_df = compare_df.dropna(subset=["comparison_daily_return_clean"]).copy()

        if not returns_plot_df.empty:
            fig_returns = px.box(
                returns_plot_df,
                x=ticker_col,
                y="comparison_daily_return_clean",
                color=ticker_col,
                title=f"Daily Return Distribution (Cleaned, {comparison_period})"
            )
            st.plotly_chart(fig_returns, use_container_width=True)
        else:
            st.info("Daily returns comparison could not be generated for the selected period.")

        st.markdown("#### 📋 Comparison Metrics")
        metrics = (
            compare_df.groupby(ticker_col)
            .agg(
                avg_return=("comparison_daily_return_clean", "mean"),
                volatility=("comparison_daily_return_clean", "std"),
                total_return=(close_col, lambda s: (s.iloc[-1] / s.iloc[0] - 1) if len(s) > 1 and pd.notna(s.iloc[0]) and s.iloc[0] != 0 else np.nan),
                latest_price=(close_col, "last"),
                observations=(close_col, "count"),
            )
            .reset_index()
            .sort_values("total_return", ascending=False)
        )
        metrics["risk_level"] = metrics["volatility"].apply(classify_risk)
        st.dataframe(metrics, use_container_width=True)

    st.markdown("---")
    st.subheader("🧠 EDA Market Insight")

    eda_left, eda_right = st.columns(2)

    with eda_left:
        st.markdown("### 📊 Market Summary")
        if not eda_market_summary_lines.empty and "summary_line" in eda_market_summary_lines.columns:
            for line in eda_market_summary_lines["summary_line"]:
                st.write(f"- {line}")
        elif not eda_market_behaviour.empty:
            st.dataframe(eda_market_behaviour, use_container_width=True)
        else:
            st.info("EDA market summary not found.")

        st.markdown("### 🧠 Core Market Insight")
        if not eda_interpretation_lines.empty and "summary_line" in eda_interpretation_lines.columns:
            for line in eda_interpretation_lines["summary_line"]:
                st.write(f"- {line}")
        elif not eda_market_interpretation.empty and "insight" in eda_market_interpretation.columns:
            for line in eda_market_interpretation["insight"]:
                st.write(f"- {line}")
        else:
            st.info("EDA interpretation not found.")

    with eda_right:
        st.markdown("### 🚀 Trading Implications")
        if not eda_trading_implications.empty and "category" in eda_trading_implications.columns and "strategy" in eda_trading_implications.columns:
            col_a, col_b = st.columns(2)

            with col_a:
                st.write("✅ Favors")
                favours = eda_trading_implications[eda_trading_implications["category"] == "favors"]
                if not favours.empty:
                    for item in favours["strategy"]:
                        st.write(f"- {item}")

            with col_b:
                st.write("⚠️ Avoid")
                avoids = eda_trading_implications[eda_trading_implications["category"] == "avoid"]
                if not avoids.empty:
                    for item in avoids["strategy"]:
                        st.write(f"- {item}")
        else:
            st.info("EDA trading implications not found.")

        st.markdown("### 📈 Top Active Stocks (Liquidity)")
        if not eda_top_active_stocks.empty:
            st.dataframe(eda_top_active_stocks, use_container_width=True)
        else:
            st.info("Top active stocks data not found.")

    extra_left, extra_right = st.columns(2)

    with extra_left:
        st.markdown("### 🚀 Top Gainers")
        if not eda_top_gainers.empty:
            st.dataframe(eda_top_gainers, use_container_width=True)
        else:
            st.info("Top gainers data not found.")

    with extra_right:
        st.markdown("### 📉 Top Losers")
        if not eda_top_losers.empty:
            st.dataframe(eda_top_losers, use_container_width=True)
        else:
            st.info("Top losers data not found.")

    eda_top_active_chart = SCREENSHOT_DIR / "eda_top_active_stocks.png"
    if eda_top_active_chart.exists():
        st.image(str(eda_top_active_chart), caption="Top Active Stocks", use_container_width=True)

    with st.expander("View Raw Signals Data"):
        st.dataframe(df.head(200), use_container_width=True)

# =========================================================
# PORTFOLIO
# =========================================================
with tab3:
    st.subheader("Portfolio")

    if dynamic_portfolio_returns is not None and not dynamic_portfolio_returns.empty:
        portfolio_growth = portfolio_returns_with_diag.copy().sort_values("rebalance_date")

        st.metric(
            "Net Total Return %",
            f"{portfolio_diagnostics.get('net_total_return_pct', 0):.2f}%"
        )
        st.caption(
            f"Rebalance Frequency: Weekly | "
            f"Universe Filter: BUY-only | "
            f"Momentum Threshold: {momentum_threshold} | "
            f"Backtest Top N: {dynamic_top_n} | "
            f"Replacement Buffer: {replacement_buffer} | "
            f"Confidence Threshold: {confidence_threshold} | "
            f"Transaction Cost: {transaction_cost:.4f} | "
            f"Minimum Holding Period: {min_holding_period} week(s)"
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Average Period Return", f"{portfolio_diagnostics.get('avg_period_return', np.nan):.4f}")
        m2.metric("Portfolio Volatility", f"{portfolio_diagnostics.get('volatility', np.nan):.4f}")
        m3.metric("Win Rate", f"{portfolio_diagnostics.get('win_rate_pct', np.nan):.1f}%")
        m4.metric(
            "Sharpe Ratio",
            f"{portfolio_diagnostics.get('sharpe_ratio', np.nan):.4f}"
            if pd.notna(portfolio_diagnostics.get("sharpe_ratio", np.nan)) else "N/A"
        )

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Max Drawdown", f"{portfolio_diagnostics.get('max_drawdown_pct', np.nan):.2f}%")
        r2.metric("Best Period", f"{portfolio_diagnostics.get('best_period_return', np.nan):.4f}")
        r3.metric("Worst Period", f"{portfolio_diagnostics.get('worst_period_return', np.nan):.4f}")
        r4.metric(
            "Gross Total Return",
            f"{portfolio_diagnostics.get('gross_total_return_pct', np.nan):.2f}%"
            if pd.notna(portfolio_diagnostics.get("gross_total_return_pct", np.nan)) else "N/A"
        )

        t1, t2 = st.columns(2)
        avg_turnover = portfolio_diagnostics.get("avg_turnover_rate", np.nan)
        max_turnover = portfolio_diagnostics.get("max_turnover_rate", np.nan)

        t1.metric(
            "Average Turnover",
            f"{avg_turnover:.2f}" if pd.notna(avg_turnover) else "N/A"
        )
        t2.metric(
            "Max Turnover",
            f"{max_turnover:.2f}" if pd.notna(max_turnover) else "N/A"
        )

        turnover_input = avg_turnover if pd.notna(avg_turnover) else 0
        momentum_style, signal_quality, holding_style, turnover_style = classify_strategy(
            momentum_threshold,
            confidence_threshold,
            min_holding_period,
            turnover_input
        )

        st.markdown("### 🧠 Strategy Profile")
        sp1, sp2 = st.columns(2)

        with sp1:
            st.write(f"**Momentum Style:** {momentum_style}")
            st.write(f"**Signal Quality:** {signal_quality}")

        with sp2:
            st.write(f"**Holding Style:** {holding_style}")
            st.write(f"**Turnover Level:** {turnover_style}")

        fig_portfolio = px.line(
            portfolio_growth,
            x="rebalance_date",
            y="equity_curve",
            title=f"Dynamic Equal-Weight Portfolio Equity Curve (Weekly Rebalance, Top {dynamic_top_n})"
        )
        st.plotly_chart(fig_portfolio, use_container_width=True)

        if "gross_portfolio_return" in portfolio_growth.columns:
            portfolio_growth["gross_equity_curve"] = (1 + portfolio_growth["gross_portfolio_return"].fillna(0)).cumprod()
            fig_gross_net = px.line(
                portfolio_growth,
                x="rebalance_date",
                y=["equity_curve", "gross_equity_curve"],
                title=f"Net vs Gross Equity Curve (Weekly Rebalance, Top {dynamic_top_n})"
            )
            st.plotly_chart(fig_gross_net, use_container_width=True)

        st.markdown("### Portfolio Return History")
        st.dataframe(portfolio_growth.tail(50), use_container_width=True)

        if not turnover_df.empty:
            st.markdown("### Turnover History")
            st.dataframe(turnover_df.tail(50), use_container_width=True)

        if dynamic_portfolio_holdings is not None and not dynamic_portfolio_holdings.empty:
            st.markdown("### Dynamic Portfolio Holdings Preview")
            holdings_display_cols = [
                c for c in [
                    "rebalance_date",
                    ticker_col,
                    signal_col,
                    momentum_col,
                    "confidence_score",
                    "ranking_score",
                    "portfolio_weight",
                    "forward_return",
                    "gross_weighted_forward_return",
                    "weighted_forward_return"
                ] if c in dynamic_portfolio_holdings.columns
            ]
            st.dataframe(dynamic_portfolio_holdings[holdings_display_cols].head(50), use_container_width=True)
        else:
            st.info("No dynamic portfolio holdings available.")

        st.markdown("---")
        st.markdown("### Step 5B — Parameter Optimization Lab")
        st.caption("Test multiple parameter combinations automatically and rank the best-performing setups.")

        run_optimization = st.checkbox("Run optimization grid", value=False, key="run_optimization_grid")

        if run_optimization:
            optimization_results = run_parameter_optimization(
                df=df,
                ticker_col=ticker_col,
                date_col=date_col,
                close_col=close_col,
                momentum_col=momentum_col,
                volatility_col=volatility_col,
                forecast_col=forecast_col,
                signal_col=signal_col,
                replacement_buffer=replacement_buffer,
                confidence_threshold=confidence_threshold,
                transaction_cost=transaction_cost,
                momentum_values=[2.0, 3.0, 4.0, 5.0],
                holding_values=[1, 2, 3, 4],
                top_n_values=[3, 5, 7],
                rebalance_frequency=rebalance_frequency,
                buy_only=buy_only_mode
            )

            if not optimization_results.empty:
                st.dataframe(optimization_results, use_container_width=True)

                top_result = optimization_results.iloc[0]

                o1, o2, o3 = st.columns(3)
                o1.metric("Best Momentum Threshold", f"{top_result['momentum_threshold']:.1f}")
                o2.metric("Best Holding Period", f"{int(top_result['min_holding_period'])} week(s)")
                o3.metric("Best Top N", int(top_result["top_n"]))

                o4, o5, o6 = st.columns(3)
                o4.metric("Best Net Return", f"{top_result['net_total_return_pct']:.2f}%")
                o5.metric("Best Sharpe", f"{top_result['sharpe_ratio']:.4f}")
                o6.metric("Best Max Drawdown", f"{top_result['max_drawdown_pct']:.2f}%")
            else:
                st.info("No optimization results could be generated.")

    else:
        st.warning("Dynamic portfolio data not available.")

# =========================================================
# AI CHAT
# =========================================================
with tab4:
    st.subheader("AI Chat")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": (
                    "Ask about BUY opportunities, best performers, market regime, "
                    "active stocks, risk, or compare two stocks."
                )
            }
        ]

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    prompt = st.chat_input("Ask about NGX market...")

    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        reply = answer_ai_query(
            prompt,
            latest_snapshot,
            summary,
            eda_market_interpretation,
            eda_trading_implications,
            eda_top_active_stocks
        )

        st.session_state.chat_history.append({"role": "assistant", "content": reply})

        with st.chat_message("assistant"):
            st.write(reply)

# =========================================================
# QUANT CONTROL CENTER
# =========================================================
with tab5:
    st.subheader("Quant Control Center")
    st.caption("Roadmap, system capability map, phase tracking, and quant build queue")

    st.markdown("## 1. System Capability Map")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
### ✅ What this data can support
- Momentum strategies
- Mean reversion strategies
- Moving-average systems
- Ranking systems
- Regime filters
- Portfolio simulations
- Cross-stock comparison
- Risk analytics
- ML classification experiments
        """)

    with c2:
        st.markdown("""
### ⚠️ What this data cannot fully support yet
- High-frequency trading (HFT)
- Order book modelling
- Execution algorithms
- Market-making systems
- Deep intraday microstructure research
- Precision transaction-cost modelling
        """)

    st.markdown("## 2. Quant Evolution Tracker")
    evolution_df = pd.DataFrame({
        "Level": [
            "Level 1 — Signal Dashboard",
            "Level 2 — Quant Research Product",
            "Level 3 — AI Trading Assistant",
            "Level 4 — NGX Quant Intelligence Platform"
        ],
        "Status": [
            "✅ Done",
            "🔄 In Progress",
            "⏳ Pending",
            "⏳ Pending"
        ],
        "Focus": [
            "Signals, charts, insights",
            "Ranking, portfolio simulation, risk metrics",
            "Explain signals, compare stocks, suggest allocations",
            "Factor models, walk-forward testing, sector rotation"
        ]
    })
    st.dataframe(evolution_df, use_container_width=True)

    st.markdown("## 3. Strategy Lab")
    strategy_lab_df = pd.DataFrame({
        "Strategy": [
            "Momentum",
            "Mean Reversion",
            "Moving Average Crossover",
            "Ranking System",
            "Regime Filter",
            "Portfolio Rotation"
        ],
        "Status": [
            "🔄 Testing",
            "⏳ Pending",
            "⏳ Pending",
            "🔄 Starting Now",
            "⏳ Pending",
            "⏳ Pending"
        ],
        "Comment": [
            "Signals and momentum fields already exist",
            "Can be built after ranking logic",
            "Good benchmark strategy",
            "Quant Phase 1 core build",
            "Will help avoid weak market periods",
            "Needed for top-5 / top-10 allocation systems"
        ]
    })
    st.dataframe(strategy_lab_df, use_container_width=True)

    st.markdown("## 4. Current Quant Phase")
    st.info(
        "We are now entering **Quant Phase 1**: "
        "build a ranking engine, stock ranking table, portfolio simulator, and risk metrics."
    )

    phase_df = pd.DataFrame({
        "Quant Phase 1 Task": [
            "Ranking score",
            "Stock ranking table",
            "Portfolio simulation",
            "Risk metrics",
            "Minimum holding period rule",
            "Parameter optimization grid"
        ],
        "Status": [
            "✅ Step 1A Done",
            "✅ Step 1B Done",
            "✅ Step 1C Done",
            "✅ Step 1D Done",
            "✅ Step 5A Done",
            "✅ Step 5B Done"
        ]
    })
    st.dataframe(phase_df, use_container_width=True)

    st.markdown("## 5. Quant Thinking Panel")
    st.warning(
        "Quant mindset = stop asking 'does the chart look good?' and start asking:\n\n"
        "- Does this strategy outperform over time?\n"
        "- What is the average return after a BUY signal?\n"
        "- What is the drawdown?\n"
        "- Does it work across stocks or only one?\n"
        "- Does it survive out-of-sample testing?"
    )

    st.markdown("## 6. Immediate Build Queue")
    build_queue_df = pd.DataFrame({
        "Order": [1, 2, 3, 4, 5],
        "Next Build": [
            "Create ranking score",
            "Rank stocks daily",
            "Select top-N equal-weight portfolio",
            "Track dynamic portfolio return over time",
            "Optimize parameter combinations automatically"
        ],
        "Why it matters": [
            "Turns signals into comparable opportunities",
            "Creates decision engine",
            "Creates investable system",
            "Creates quant-style portfolio simulation",
            "Moves from guesswork to systematic testing"
        ]
    })
    st.dataframe(build_queue_df, use_container_width=True)

    st.markdown("## 7. Product Destination")
    product_df = pd.DataFrame({
        "Path": [
            "Portfolio Project",
            "Sellable Tool",
            "SaaS MVP",
            "Quant Intelligence Platform"
        ],
        "What it means": [
            "Strong analytics + trading system project",
            "NGX stock signal / ranking dashboard",
            "AI assistant for African markets",
            "Full quant research and allocation platform"
        ],
        "Current Fit": [
            "✅ Already strong",
            "✅ Realistic next stage",
            "🔄 Possible with more layers",
            "⏳ Long-term build"
        ]
    })
    st.dataframe(product_df, use_container_width=True)

    st.markdown("## 8. Portfolio Candidates — Step 1B")
    st.caption("Top ranked stocks selected into a simple equal-weight model portfolio.")

    portfolio_top_n = st.slider(
        "Select number of stocks for equal-weight portfolio",
        min_value=3,
        max_value=10,
        value=5,
        key="portfolio_top_n"
    )

    dynamic_portfolio_candidates = build_equal_weight_portfolio(
        latest_snapshot=latest_snapshot,
        top_n=portfolio_top_n,
        ticker_col=ticker_col,
        close_col=close_col,
        signal_col=signal_col,
        momentum_col=momentum_col,
        buy_only_mode=True,
        momentum_threshold=momentum_threshold,
        confidence_threshold=confidence_threshold
    )

    if not dynamic_portfolio_candidates.empty:
        st.dataframe(dynamic_portfolio_candidates, use_container_width=True)

        avg_ranking_score = dynamic_portfolio_candidates["ranking_score"].mean() if "ranking_score" in dynamic_portfolio_candidates.columns else np.nan
        buy_count = (
            (dynamic_portfolio_candidates[signal_col].astype(str).str.upper() == "BUY").sum()
            if signal_col and signal_col in dynamic_portfolio_candidates.columns else 0
        )
        avg_confidence = dynamic_portfolio_candidates["confidence_score"].mean() if "confidence_score" in dynamic_portfolio_candidates.columns else np.nan

        pc1, pc2, pc3 = st.columns(3)
        pc1.metric("Selected Stocks", len(dynamic_portfolio_candidates))
        pc2.metric("Average Ranking Score", f"{avg_ranking_score:.2f}" if pd.notna(avg_ranking_score) else "N/A")
        pc3.metric("BUY Signals in Portfolio", int(buy_count))

        if pd.notna(avg_confidence):
            st.metric("Average Confidence", f"{avg_confidence:.0f}/100")

        fig_portfolio_weights = px.bar(
            dynamic_portfolio_candidates,
            x=ticker_col,
            y="portfolio_weight_pct",
            title=f"Equal-Weight Portfolio Allocation (Top {portfolio_top_n})"
        )
        st.plotly_chart(fig_portfolio_weights, use_container_width=True)
    else:
        st.info("Portfolio candidates could not be generated.")