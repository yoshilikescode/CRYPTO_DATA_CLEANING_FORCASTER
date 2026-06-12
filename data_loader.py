"""
data_loader.py
Handles loading cryptocurrency price data from CSV files or live sources.
"""
import pandas as pd
import glob
import os
import yfinance as yf


def load_live_crypto_data(start_date, end_date):
    """
    Loads live cryptocurrency data from Yahoo Finance.
    Same output format as load_crypto_data() so the
    rest of the pipeline works identically.

    Args:
        start_date: start date string e.g. "2019-01-01"
        end_date:   end date string e.g. "2024-01-01"

    Returns:
        DataFrame with dates as index, one column per coin
    """
    # Yahoo Finance ticker symbols for our 8 coins
    # Note: FTX collapsed so we replace it with Solana
    coins = {
        "btc": "BTC-USD",
        "eth": "ETH-USD",
        "bnb": "BNB-USD",
        "xrp": "XRP-USD",
        "doge": "DOGE-USD",
        "usdt": "USDT-USD",
        "usdc": "USDC-USD",
        "sol": "SOL-USD"  # replaced FTX with Solana
    }

    dfs = []
    for name, ticker in coins.items():
        print(f"Fetching {name}...")

        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False
        )

        # yfinance sometimes returns multi-level columns
        # Flatten them before renaming
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[["Close"]].rename(columns={"Close": name})
        dfs.append(df)

    #Combine

    combined = pd.concat(dfs, axis=1)

    # Force ALL column names to be simple strings
    # This handles any yfinance multi-level column weirdness
    combined.columns = [
        col if isinstance(col, str) else col[-1]
        for col in combined.columns
    ]

    # Fill weekend gaps
    combined = combined.ffill()

    print(f"Live data loaded: {combined.shape}")
    print(f"Columns: {combined.columns.tolist()}")
    return combined







def load_crypto_data(folder_path, start_date, end_date):
    """
    Loads closing prices for all cryptocurrencies in a folder.

    Args:
        folder_path: path pattern to CSV files e.g. "data/coins/*.csv"
        start_date:  earliest date to include e.g. "2019-01-01"
        end_date:    latest date to include e.g. "2022-11-15"

    Returns:
        DataFrame with dates as index, one column per cryptocurrency
    """
    # Find all CSV files matching the path pattern
    files = glob.glob(folder_path)

    # Create a complete date range as our base table
    # This ensures no dates are missing even if CSVs have gaps
    dates = pd.DataFrame(
        pd.date_range(start=start_date, end=end_date, freq="D"),
        columns=["Date"]
    )

    # Load each coin's data and attach to our date table
    for file in files:
        # Only read Date and Close columns to save memory
        df = pd.read_csv(file, usecols=["Date", "Close"])

        # Convert date strings to proper datetime objects
        df["Date"] = pd.to_datetime(df["Date"])

        # Left merge keeps ALL dates even if coin has no data that day
        dates = dates.merge(df, how="left", on="Date")

        # Rename Close column to the coin's name (e.g. "bitcoin")
        coin_name = os.path.basename(file).replace(".csv", "")
        dates = dates.rename(columns={"Close": coin_name})

    # Make Date the row index (standard for time series)
    dates.set_index("Date", inplace=True)
    dates = dates.sort_index()

    return dates
