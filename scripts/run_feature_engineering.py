from pathlib import Path
import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "outputs" / "cleaned_stock_data.csv"
OUTPUT_PATH = BASE_DIR / "outputs" / "technical_indicators.csv"


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate technical indicators and features for each stock ticker.
    """

    df = df.copy()

    # Make sure data is sorted correctly before feature creation
    df = df.sort_values(by=["ticker", "date"]).reset_index(drop=True)

    # Treat zero or negative close prices as invalid for modelling
    df.loc[df["close"] <= 0, "close"] = np.nan

    # Group by ticker so each stock is calculated separately
    grouped = df.groupby("ticker")

    # --- Basic Returns ---
    df["daily_return"] = grouped["close"].pct_change()

    df["log_return"] = grouped["close"].transform(
        lambda x: np.where((x > 0) & (x.shift(1) > 0), np.log(x / x.shift(1)), np.nan)
    )

    # --- Moving Averages ---
    df["sma_20"] = grouped["close"].transform(lambda x: x.rolling(20).mean())
    df["sma_50"] = grouped["close"].transform(lambda x: x.rolling(50).mean())
    df["sma_200"] = grouped["close"].transform(lambda x: x.rolling(200).mean())

    # --- Volatility ---
    df["volatility_20"] = df.groupby("ticker")["daily_return"].transform(
        lambda x: x.rolling(20).std()
    )

    # --- Momentum ---
    df["momentum_10"] = grouped["close"].transform(lambda x: x - x.shift(10))

    # --- EMA ---
    df["ema_20"] = grouped["close"].transform(
        lambda x: x.ewm(span=20, adjust=False).mean()
    )

    return df


def main():
    print("Loading cleaned data...")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    print(f"Input shape: {df.shape}")

    df = calculate_features(df)
    print(f"Feature dataset shape: {df.shape}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("\nTechnical indicators saved successfully.")
    print(f"Output saved to: {OUTPUT_PATH}")

    print("\nPreview:")
    print(df.head())


if __name__ == "__main__":
    main()