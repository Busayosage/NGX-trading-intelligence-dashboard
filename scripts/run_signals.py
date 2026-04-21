from pathlib import Path
import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "outputs" / "forecast_output.csv"
OUTPUT_PATH = BASE_DIR / "outputs" / "signal_output.csv"


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate trend-confirmation buy/hold/sell signals.
    """

    df = df.copy()

    # Trend confirmation signal:
    # BUY when short-term trend is above medium-term trend
    # and price is still above the short-term trend
    # SELL when short-term trend falls below medium-term trend
    # Otherwise HOLD
    df["signal_combined"] = np.where(
        (df["sma_20"] > df["sma_50"]) &
        (df["close"] > df["sma_20"]),
        "BUY",

        np.where(
            (df["sma_20"] < df["sma_50"]),
            "SELL",
            "HOLD"
        )
    )

    return df


def main():
    print("Loading forecast data...")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    print(f"Input shape: {df.shape}")

    df = generate_signals(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("\nTrend-confirmation signals generated.")
    print(f"Output saved to: {OUTPUT_PATH}")

    print("\nPreview:")
    print(df[["ticker", "date", "close", "sma_20", "sma_50", "signal_combined"]].head())


if __name__ == "__main__":
    main()