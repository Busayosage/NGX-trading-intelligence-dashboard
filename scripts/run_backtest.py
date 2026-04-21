from pathlib import Path
import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "outputs" / "signal_output.csv"
OUTPUT_PATH = BASE_DIR / "outputs" / "backtest_results.csv"
PORTFOLIO_OUTPUT_PATH = BASE_DIR / "outputs" / "portfolio_backtest_results.csv"


def run_backtest(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()

    # Sort correctly
    df = df.sort_values(by=["ticker", "date"]).reset_index(drop=True)

    # Convert signals into positions
    df["position"] = np.where(
        df["signal_combined"] == "BUY", 1,
        np.where(df["signal_combined"] == "SELL", 0, np.nan)
    )

    # Carry forward positions
    df["position"] = df.groupby("ticker")["position"].ffill().fillna(0)

    # Shift for execution
    df["position_shifted"] = df.groupby("ticker")["position"].shift(1).fillna(0)

    # Strategy return (with position sizing)
    df["strategy_return"] = df["position_shifted"] * 0.3 * df["daily_return"]

    # Risk control
    df["strategy_return"] = df["strategy_return"].clip(lower=-0.05, upper=0.05)

    df["strategy_return"] = df["strategy_return"].fillna(0)

    # Portfolio aggregation
    portfolio_df = (
        df.groupby("date", as_index=False)["strategy_return"]
        .mean()
        .rename(columns={"strategy_return": "portfolio_daily_return"})
    )

    portfolio_df["portfolio_cumulative_return"] = (
        1 + portfolio_df["portfolio_daily_return"]
    ).cumprod()

    return df, portfolio_df


def calculate_metrics(df: pd.DataFrame, portfolio_df: pd.DataFrame) -> None:
    total_return = portfolio_df["portfolio_cumulative_return"].iloc[-1] - 1

    trades = (
        df.groupby("ticker")["position_shifted"]
        .diff()
        .abs()
        .fillna(0)
        .sum()
    )

    win_rate = (portfolio_df["portfolio_daily_return"] > 0).mean()

    num_days = len(portfolio_df)
    annualised_return = (1 + total_return) ** (252 / num_days) - 1 if num_days > 0 else 0

    daily_mean = portfolio_df["portfolio_daily_return"].mean()
    daily_std = portfolio_df["portfolio_daily_return"].std()
    sharpe_ratio = (daily_mean / daily_std) * np.sqrt(252) if daily_std != 0 else 0

    cumulative = portfolio_df["portfolio_cumulative_return"]
    running_peak = cumulative.cummax()
    drawdown = (cumulative - running_peak) / running_peak
    max_drawdown = drawdown.min()

    print("\nBacktest Results:")
    print(f"Total Return: {total_return:.2%}")
    print(f"Annualised Return: {annualised_return:.2%}")
    print(f"Number of Trades: {int(trades)}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")


def main():
    print("Loading signal data...")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    print(f"Input shape: {df.shape}")

    stock_df, portfolio_df = run_backtest(df)

    calculate_metrics(stock_df, portfolio_df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    stock_df.to_csv(OUTPUT_PATH, index=False)
    portfolio_df.to_csv(PORTFOLIO_OUTPUT_PATH, index=False)

    print("\nBacktest results saved.")
    print(f"Stock-level output: {OUTPUT_PATH}")
    print(f"Portfolio-level output: {PORTFOLIO_OUTPUT_PATH}")

    print("\nStock-level preview:")
    print(
        stock_df[
            ["ticker", "date", "signal_combined", "strategy_return", "position_shifted"]
        ].head()
    )

    print("\nPortfolio-level preview:")
    print(portfolio_df.head())


if __name__ == "__main__":
    main()