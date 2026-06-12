"""
main.py
Entry point for the Crypto Portfolio Forecasting project.

Pipeline:
1. Load cryptocurrency price data
2. Clean and prepare data
3. Build equal-weighted portfolio index
4. Load external economic factors
5. Filter factors by correlation strength
6. Split into training and testing sets
7. Train and evaluate ARIMA model
8. Train and evaluate SARIMAX model
"""
import os
from data_loader import load_crypto_data
from data_cleaner import clean_crypto_data
from index_builder import daily_index_builder, prepare_model_data
from visualizer import plot_index
from factors_loader import load_factors
from forecasting_models import split_data, run_arima, run_sarimax
import warnings
import pandas as pd


# Suppress statsmodels frequency warnings
warnings.filterwarnings("ignore")



# Display full DataFrames in terminal output
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

# ── PROJECT PATHS ─────────────────────────────────────────────
base_dir = os.path.dirname(os.path.abspath(__file__))
coins_path = os.path.join(
    base_dir, "CryptoCurrency", "data",
    "cryptocurrency", "coins", "*.csv"
)

# ── STEP 1: LOAD ──────────────────────────────────────────────
print("=" * 50)
print("STEP 1: Loading cryptocurrency data...")
print("=" * 50)
cryptos = load_crypto_data(
    folder_path=coins_path,
    start_date="2019-01-01",
    end_date="2022-11-15"
)
print(f"Loaded {cryptos.shape[1]} coins, {cryptos.shape[0]} days\n")

# ── STEP 2: CLEAN ─────────────────────────────────────────────
print("=" * 50)
print("STEP 2: Cleaning data...")
print("=" * 50)
cryptos_clean, start_date, end_date = clean_crypto_data(cryptos)

# ── STEP 3: BUILD INDEX ───────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 3: Building portfolio index...")
print("=" * 50)
index = daily_index_builder(cryptos_clean)

# ── STEP 4: VISUALIZE INDEX ───────────────────────────────────
print("\n" + "=" * 50)
print("STEP 4: Plotting portfolio index...")
print("=" * 50)
plot_index(index)

# ── STEP 5: LOAD EXTERNAL FACTORS ────────────────────────────
print("\n" + "=" * 50)
print("STEP 5: Loading external factors...")
print("=" * 50)
factors = load_factors(base_dir, start_date, end_date)

# ── STEP 6: CORRELATION FILTERING ────────────────────────────
print("\n" + "=" * 50)
print("STEP 6: Finding correlating factors...")
print("=" * 50)
data_prepared = prepare_model_data(index, factors)

# ── STEP 7: SPLIT DATA ────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 7: Splitting into train/test sets...")
print("=" * 50)
X_train, X_test, y_train, y_test = split_data(data_prepared)

# ── STEP 8: ARIMA MODEL ───────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 8: Running ARIMA model...")
print("=" * 50)
arima_pred, arima_rmse = run_arima(y_train, y_test)

# ── STEP 9: SARIMAX MODEL ─────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 9: Running SARIMAX model...")
print("=" * 50)
sarimax_pred, sarimax_rmse = run_sarimax(
    X_train, X_test, y_train, y_test
)

# ── RESULTS SUMMARY ───────────────────────────────────────────
print("\n" + "=" * 50)
print("RESULTS SUMMARY")
print("=" * 50)
print(f"ARIMA  RMSE: {arima_rmse:.6f}")
print(f"SARIMAX RMSE: {sarimax_rmse:.6f}")
print(f"Winner: {'ARIMA' if arima_rmse < sarimax_rmse else 'SARIMAX'}")

