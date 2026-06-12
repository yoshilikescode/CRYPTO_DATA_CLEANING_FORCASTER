"""
data_cleaner.py
Handles cleaning and preparing raw cryptocurrency data for analysis.
"""
import pandas as pd


def find_start_date(df):
    """
    Finds the first date where ALL cryptocurrencies have price data.
    This is determined by the coin with the most missing values (FTX).

    Args:
        df: raw cryptocurrency DataFrame

    Returns:
        tuple: (start_date, end_date)
    """
    # Count missing values per coin and find which has the most
    missing_values = df.isnull().sum()
    max_missing_col = missing_values.idxmax()

    # Find first date that coin has actual data
    start_date = df[max_missing_col].first_valid_index()
    end_date = df.index[-1]

    print(f"Start date: {start_date}")
    print(f"End date: {end_date}")

    return start_date, end_date


def clean_crypto_data(df):
    """
    Cleans cryptocurrency data by:
    1. Removing dates before all coins existed
    2. Filling weekend/holiday gaps with previous day's value

    Args:
        df: raw cryptocurrency DataFrame

    Returns:
        tuple: (cleaned DataFrame, start_date, end_date)
    """
    start_date, end_date = find_start_date(df)

    # Remove all dates before FTX launched
    df_clean = df[start_date:]

    # Fill gaps (weekends/holidays) with previous day's value
    # This is called "forward fill" - carries last known value forward
    df_clean = df_clean.ffill()

    print(f"Shape after cleaning: {df_clean.shape}")
    print(f"Missing values after cleaning:\n{df_clean.isnull().sum()}")

    return df_clean, start_date, end_date