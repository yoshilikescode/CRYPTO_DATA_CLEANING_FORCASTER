"""
factors_loader.py
Loads external economic factors that may influence cryptocurrency prices.

Factors included:
- US Treasury interest rates (daily)
- Consumer Price Index / Personal Saving Rate (monthly)
- Stock market indices: S&P500, Nasdaq (daily)
- Commodity metals: Gold, Silver (daily)
"""
import pandas as pd
import glob
import os


def load_factors(base_dir, start_date, end_date):
    """
    Loads and combines all external economic factors into one table.

    Monthly data (CPI, PSR) is forward-filled to daily frequency.
    Weekend/holiday gaps are filled with previous day's values.

    Args:
        base_dir:   root project directory
        start_date: first date to include
        end_date:   last date to include

    Returns:
        DataFrame with dates as index, one column per factor
    """

    def build_path(folder):
        """Helper to build consistent folder paths"""
        return os.path.join(
            base_dir, "CryptoCurrency", "data",
            "cryptocurrency", folder, "*.csv"
        )

    # ── INTEREST RATES ────────────────────────────────────────
    # Multiple yearly files stacked vertically into one table
    rate_files = glob.glob(build_path("interest_rate"))
    rate_dfs = []
    for f in rate_files:
        df = pd.read_csv(f)
        df["Date"] = pd.to_datetime(df["Date"])
        rate_dfs.append(df)

    # Stack all years on top of each other (vertical combine)
    rates = pd.concat(rate_dfs, ignore_index=True)
    rates.set_index("Date", inplace=True)
    rates = rates.sort_index()

    # ── ECONOMIC INDEX ────────────────────────────────────────
    # CPI and PSR are monthly - will be forward filled to daily
    econ_files = glob.glob(build_path("economic_index"))
    econ_dfs = []
    for f in econ_files:
        df = pd.read_csv(f, usecols=["Date", "Rate"])
        df["Date"] = pd.to_datetime(df["Date"])

        # Rename Rate column to the file name (e.g. "consumer-price-index")
        name = os.path.basename(f).replace(".csv", "")
        df = df.rename(columns={"Rate": name})
        df.set_index("Date", inplace=True)
        econ_dfs.append(df)

    # Combine side by side (CPI | PSR)
    economic = pd.concat(econ_dfs, axis=1)

    # ── STOCK INDICES ─────────────────────────────────────────
    # S&P500 and Nasdaq closing prices only
    stock_files = glob.glob(build_path("stock_index"))
    stock_dfs = []
    for f in stock_files:
        df = pd.read_csv(f, usecols=["Date", "Price"])
        df["Date"] = pd.to_datetime(df["Date"])
        name = os.path.basename(f).replace(".csv", "")
        df = df.rename(columns={"Price": name})
        df.set_index("Date", inplace=True)
        stock_dfs.append(df)

    stocks = pd.concat(stock_dfs, axis=1)

    # ── METALS ────────────────────────────────────────────────
    # Gold and Silver closing prices only
    metal_files = glob.glob(build_path("metals"))
    metal_dfs = []
    for f in metal_files:
        df = pd.read_csv(f, usecols=["Date", "Price"])
        df["Date"] = pd.to_datetime(df["Date"])
        name = os.path.basename(f).replace(".csv", "")
        df = df.rename(columns={"Price": name})
        df.set_index("Date", inplace=True)
        metal_dfs.append(df)

    metals = pd.concat(metal_dfs, axis=1)

    # ── COMBINE ALL FACTORS ───────────────────────────────────
    all_factors = pd.concat(
        [rates, economic, stocks, metals],
        axis=1
    )

    # Filter to our date range
    all_factors = all_factors[start_date:end_date]

    # Forward fill gaps (weekends, holidays, monthly data)
    # Then backward fill any remaining NaN at the start
    all_factors = all_factors.ffill().bfill()

    print(f"Factors shape: {all_factors.shape}")
    print(f"Factors columns: {all_factors.columns.tolist()}")

    return all_factors