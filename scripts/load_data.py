from pathlib import Path
import sqlite3
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "stock_data.csv"
DATABASE_PATH = BASE_DIR / "database" / "market_data.db"
OUTPUT_PATH = BASE_DIR / "outputs" / "cleaned_stock_data.csv"


def load_and_validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean long-format stock data.
    Expected columns include:
    ticker_clean OR ticker + date + close
    """

    # Standardise column names
    df.columns = df.columns.str.strip().str.lower()

    # Handle ticker column variations
    if "ticker" not in df.columns:
        if "ticker_clean" in df.columns:
            df.rename(columns={"ticker_clean": "ticker"}, inplace=True)
        elif "company" in df.columns:
            df.rename(columns={"company": "ticker"}, inplace=True)
        elif "stock" in df.columns:
            df.rename(columns={"stock": "ticker"}, inplace=True)
        elif "name" in df.columns:
            df.rename(columns={"name": "ticker"}, inplace=True)
        else:
            raise ValueError("No ticker column found in dataset")

    # Check required columns
    required_cols = {"ticker", "date", "close"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Convert data types
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    # Remove invalid rows
    df = df.dropna(subset=["date", "close"])

    # Clean ticker values
    df["ticker"] = df["ticker"].str.strip().str.upper()

    # Keep only relevant columns for modelling
    df = df[["ticker", "date", "close"]]

    # Sort data
    df = df.sort_values(by=["ticker", "date"]).reset_index(drop=True)

    return df


def save_to_sqlite(df: pd.DataFrame, db_path: Path):
    """
    Save DataFrame to SQLite database
    """
    conn = sqlite3.connect(db_path)
    df.to_sql("stock_prices", conn, if_exists="replace", index=False)
    conn.close()


def main():
    print("Loading cleaned stock dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"File not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"Raw shape: {df.shape}")

    df = load_and_validate_data(df)
    print(f"Cleaned shape: {df.shape}")

    # Save cleaned CSV
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    # Save to SQLite
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_to_sqlite(df, DATABASE_PATH)

    print("\nData successfully saved.")
    print(f"CSV saved to: {OUTPUT_PATH}")
    print(f"Database saved to: {DATABASE_PATH}")

    print("\nPreview:")
    print(df.head())


if __name__ == "__main__":
    main()