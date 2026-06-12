"""
forecasting.py
Time series forecasting models for cryptocurrency portfolio index.

Models implemented:
- ARIMA: uses only historical index values
- SARIMAX: uses historical values + external economic factors

Future models can be added as new functions following
the same pattern: run_modelname(X_train, X_test, y_train, y_test)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima


def split_data(data_prepared, split_ratio=0.95):
    """
    Splits prepared data into training and testing sets.

    Unlike random ML splits, time series MUST be split
    chronologically - future data cannot train past models.

    95% training (~3 years) / 5% testing (~2 months)

    Args:
        data_prepared: normalized DataFrame with factors + Index
        split_ratio:   proportion of data used for training

    Returns:
        X_train, X_test: external factors for train/test periods
        y_train, y_test: index values for train/test periods
    """
    train_size = int(len(data_prepared) * split_ratio)

    # X = external factors (all columns except last)
    # y = index values (last column only)
    X_train = data_prepared.iloc[:train_size, :-1]
    X_test = data_prepared.iloc[train_size:, :-1]
    y_train = data_prepared.iloc[:train_size, -1:]
    y_test = data_prepared.iloc[train_size:, -1:]

    print(f"Training size: {len(y_train)} days")
    print(f"Testing size: {len(y_test)} days")

    return X_train, X_test, y_train, y_test


#""" The 3 Knobs of ARIMA1. The p Knob (Auto-Regressive / The Rearview Mirror)The p knob controls how many days into the immediate past the model looks to guess today's price.If you set $p=1$, the model looks only at yesterday's crypto price to guess today's price.If you set $p=5$, the model looks at the last 5 days of history. It assumes that if crypto has been rising for 5 straight days, it has "momentum" and might keep rising today.2. The d Knob (Integrated / The Stabilizer)Crypto data is notorious for having a "trend"—it generally crawls upward over months or crashes downward over weeks. This constant movement messes up statistical math, which prefers data that fluctuates around a steady average line.The d parameter stands for Differencing. It tells Python: "Don't try to predict the actual price today. Instead, subtract yesterday's price from today's price, and predict the change in price."If you look at absolute prices, they are a roller coaster.If you look at daily changes (e.g., $+\$15$, $-\$4$, $+\$2$), the data stabilizes into a predictable wave. Setting $d=1$ means we do this subtraction step exactly once to steady the data before modeling.3. The q Knob (Moving Average / The Shock Absorber)Markets don't just move on smooth momentum; they get hit by sudden, random "shocks" (like Elon Musk tweeting about a coin, or a sudden regulatory announcement).When a shock happens, it leaves behind an "error" (a difference between what the market normally does and what actually happened). The q knob tells the model how many days a past shock should echo into the future.If a massive shock happened yesterday, does it still affect the price today? If $q=1$, the model remembers yesterday's shock and uses it to adjust today's prediction.It essentially asks: "How long does the splash from yesterday's rock take to settle down?"What is auto_arima doing?Finding the perfect combination of these three knobs—like $(p=2, d=1, q=3)$—is incredibly tedious to do by hand.That is where auto_arima comes to the rescue. Think of auto_arima as an automated tester. You give it your raw crypto history (y_train), and it runs a rapid-fire experiment behind the scenes:It tries $(1, 1, 1)$ and grades its performance.It tries $(2, 1, 1)$ and grades its performance.It tries $(1, 1, 2)$ and grades its performance.It keeps twisting the knobs until it finds the exact combination that yields the lowest error rate. When you see this line in your script output:Best ARIMA order: (2, 1, 1)It means auto_arima is telling you: "Hey, I tested everything. This data behaves best when looking back 2 days for momentum ($p=2$), subtracting yesterday's price once ($d=1$), and remembering 1 day of random market shocks ($q=1$)." """


def run_arima(y_train, y_test):
    """
    Trains and evaluates an ARIMA model.

    ARIMA(p,d,q):
    - p: number of past values to include
    - d: differencing order (1 = first difference)
    - q: number of past error terms to include

    Uses auto_arima to find optimal p and q automatically.
    Only uses historical index values - no external factors.

    Args:
        y_train: index values for training
        y_test:  index values for testing (ground truth)

    Returns:
        Series of predicted index values
    """
    print("\nFinding best ARIMA parameters...")

    # Automatically test combinations of p and q
    # Picks the one with lowest AIC score
    arima_param = auto_arima(
        y_train,
        start_p=1, d=1, start_q=1,
        max_p=5, max_q=5,
        seasonal=False,
        trace=True
    )

    print(f"Best ARIMA order: {arima_param.order}")
    print("\nTraining ARIMA model...")

    # Train model with best parameters
    model = ARIMA(y_train, order=arima_param.order).fit()

    # Generate predictions for test period
    forecast = model.get_forecast(steps=len(y_test))
    prediction = forecast.predicted_mean
    ci = forecast.conf_int()

    # Align prediction dates with test dates
    prediction.index = y_test.index
    ci.index = y_test.index

    # RMSE: average prediction error (lower = better)
    rmse = np.sqrt(mean_squared_error(y_test, prediction))
    print(f"ARIMA RMSE: {round(rmse, 6)}")

    # Plot results
    plt.figure(figsize=(18, 6))
    plt.plot(y_test.index, y_test,
             label="Actual", color="blue")
    plt.plot(prediction.index, prediction,
             label="ARIMA Prediction", color="orange")
    plt.fill_between(ci.index,
                     ci.iloc[:, 0], ci.iloc[:, 1],
                     color="grey", alpha=0.3,
                     label="Confidence Interval")
    plt.title("ARIMA Model Prediction", fontsize=20)
    plt.xlabel("Date", fontsize=13)
    plt.ylabel("Index Value", fontsize=13)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return prediction, rmse, ci


def run_sarimax(X_train, X_test, y_train, y_test):
    """
    Trains and evaluates a SARIMAX model.

    SARIMAX extends ARIMA with:
    - Seasonality (P,D,Q,m): captures weekly crypto patterns
    - Exogenous factors: uses external data (gold, rates, stocks)

    Parameters:
    - order(p,d,q): same as ARIMA
    - seasonal_order(P,D,Q,m): seasonal equivalent, m=7 for weekly

    Args:
        X_train: external factors for training period
        X_test:  external factors for testing period
        y_train: index values for training
        y_test:  index values for testing (ground truth)

    Returns:
        Series of predicted index values
    """
    print("\nFinding best SARIMAX parameters...")

    # auto_arima with seasonal=True finds both
    # regular and seasonal parameters
    sarimax_param = auto_arima(
        y_train,
        exogenous=X_train,
        m=7,  # weekly seasonality
        start_p=0, d=1, start_q=0,
        start_P=0, D=1, start_Q=0,
        max_p=3, max_q=1,
        max_P=3, max_Q=1,
        seasonal=True,
        trace=True
    )

    print(f"Best SARIMAX order: {sarimax_param.order}")
    print(f"Best seasonal order: {sarimax_param.seasonal_order}")
    print("\nTraining SARIMAX model...")
    """ARIMA:
    "Based on how the index moved in the past,
    I predict it will do X"
    → Like predicting using only memory

    SARIMAX:
    "Based on past index movement AND the fact that
    S&P500 dropped today AND interest rates rose,
    I predict it will do X"
    → Like predicting using memory + current news"""
    # Train with both index history and external factors
    model = SARIMAX(
        endog=y_train,  # what we're predicting
        exog=X_train,  # external factors helping prediction
        order=sarimax_param.order,
        seasonal_order=sarimax_param.seasonal_order
    ).fit(disp=False, maxiter=500, method='powell')

    # Generate predictions using real external factor values
    forecast = model.get_prediction(
        start=len(y_train),
        end=len(y_train) + len(y_test) - 1,
        exog=X_test,
        dynamic=True
    )
    prediction = forecast.predicted_mean
    ci = forecast.conf_int()

    # Align prediction dates with test dates
    prediction.index = y_test.index
    ci.index = y_test.index

    # RMSE: average prediction error (lower = better)
    rmse = np.sqrt(mean_squared_error(y_test, prediction))
    print(f"SARIMAX RMSE: {round(rmse, 6)}")

    # Plot results
    plt.figure(figsize=(18, 6))
    plt.plot(y_test.index, y_test,
             label="Actual", color="blue")
    plt.plot(prediction.index, prediction,
             label="SARIMAX Prediction", color="orange")
    plt.fill_between(ci.index,
                     ci.iloc[:, 0], ci.iloc[:, 1],
                     color="grey", alpha=0.3,
                     label="Confidence Interval")
    plt.title("SARIMAX Model Prediction", fontsize=20)
    plt.xlabel("Date", fontsize=13)
    plt.ylabel("Index Value", fontsize=13)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return prediction, rmse, ci

