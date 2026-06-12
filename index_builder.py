"""
index_builder.py
Builds a custom equal-weighted index from multiple cryptocurrencies.
Similar to how S&P 500 tracks 500 companies with one number.
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def normalize_data(df):
    """
    Scales all columns to range [0,1] using Min-Max normalization.
    Formula: (value - min) / (max - min)

    This prevents high-priced coins (Bitcoin) from dominating
    lower-priced coins (Dogecoin) in our index calculation.

    Args:
        df: DataFrame with raw prices

    Returns:
        DataFrame with all values scaled between 0 and 1
    """
    scaler = MinMaxScaler()

    # fit_transform learns min/max then scales all values
    scaled_values = scaler.fit_transform(df)

    # Rebuild DataFrame preserving column names and dates
    df_normalized = pd.DataFrame(
        scaled_values,
        columns=df.columns,
        index=df.index
    )

    return df_normalized


def daily_index_builder(df):
    """
    Creates a single index value per day by averaging
    all normalized coin prices equally.

    Each coin gets weight = 1/N where N = number of coins.

    Args:
        df: cleaned cryptocurrency DataFrame

    Returns:
        DataFrame with single 'Index' column
    """
    # Normalize so all coins are on same scale
    df_normalized = normalize_data(df)

    # Average across all coins for each day
    # axis=1 means average across columns (not down rows)
    index_values = df_normalized.mean(axis=1)

    index_df = pd.DataFrame(index_values, columns=["Index"])

    return index_df


def prepare_model_data(index_df, factors_df):
    """
    Combines index and external factors, normalizes everything,
    and removes weakly correlated factors.

    Correlation threshold: factors with correlation between
    -0.2 and 0.2 are dropped (too weak to help the model)

    Args:
        index_df:   DataFrame with Index column
        factors_df: DataFrame with external economic factors

    Returns:
        Normalized DataFrame ready for model training
    """
    # Combine factors and index into one table
    data = pd.concat([factors_df, index_df], axis=1)

    # Remove rows where Index has no value
    data = data.dropna(subset=["Index"])

    # Fill any remaining gaps in factors
    data = data.ffill().bfill()

    # Normalize all columns to [0,1] scale
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    data_nor = pd.DataFrame(
        scaled,
        columns=data.columns,
        index=data.index
    )

    # Calculate how strongly each factor correlates with Index
    correlations = data_nor.corr()["Index"]
    print("\nCorrelations with Index:")
    print(correlations.sort_values(ascending=False))

    # Drop factors with weak correlation (between -0.2 and 0.2)
    weak = correlations[
        (correlations >= -0.2) &
        (correlations <= 0.2)
        ].index.tolist()

    print(f"\nDropping weak columns: {weak}")
    data_nor = data_nor.drop(columns=weak)

    # Ensure Index is always the last column
    # (required by split_data function)
    cols = [col for col in data_nor.columns if col != "Index"]
    cols.append("Index")
    data_nor = data_nor[cols]

    print(f"Final shape: {data_nor.shape}")
    return data_nor