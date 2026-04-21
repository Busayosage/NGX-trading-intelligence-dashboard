from pathlib import Path
import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "outputs" / "technical_indicators.csv"
OUTPUT_PATH = BASE_DIR / "outputs" / "forecast_output.csv"


def generate_forecasts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create baseline forecasts and direction predictions
    """

    df = df.copy()

    # Sort data
    df = df.sort_values(by=["ticker", "date"]).reset_index(drop=True)

    grouped = df.groupby("ticker")

    # --- Baseline 1: Previous Close ---
    df["forecast_prev_close"] = grouped["close"].shift(1)

    # --- Baseline 2: SMA 20 ---
    df["forecast_sma_20"] = df["sma_20"]

    # --- Actual future movement ---
    df["actual_next_close"] = grouped["close"].shift(-1)

    # --- Direction (Actual) ---
    df["actual_direction"] = np.where(
        df["actual_next_close"] > df["close"], 1, 0
    )

    # --- Direction (Prediction: Prev Close) ---
    df["pred_direction_prev_close"] = np.where(
        df["forecast_prev_close"] < df["close"], 1, 0
    )

    # --- Direction (Prediction: SMA 20) ---
    df["pred_direction_sma_20"] = np.where(
        df["forecast_sma_20"] < df["close"], 1, 0
    )

    return df


def evaluate_forecasts(df: pd.DataFrame):
    """
    Evaluate directional accuracy
    """

    df_eval = df.dropna(subset=["actual_direction"])

    accuracy_prev = (df_eval["actual_direction"] == df_eval["pred_direction_prev_close"]).mean()
    accuracy_sma = (df_eval["actual_direction"] == df_eval["pred_direction_sma_20"]).mean()

    print("\nForecast Evaluation:")
    print(f"Prev Close Accuracy: {accuracy_prev:.2%}")
    print(f"SMA 20 Accuracy: {accuracy_sma:.2%}")


def main():
    print("Loading feature dataset...")

    df = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    print(f"Input shape: {df.shape}")

    df = generate_forecasts(df)

    evaluate_forecasts(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("\nForecast output saved.")
    print(f"Output: {OUTPUT_PATH}")

    print("\nPreview:")
    print(df.head())


if __name__ == "__main__":
    main()