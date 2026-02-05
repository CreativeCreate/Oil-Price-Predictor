"""
Machine Learning App for Simple Oil Price Prediction
Predicts future closing price of crude oil (WTI) using historical data.
Uses Random Forest, Support Vector Regression, and a simple Neural Network.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Data
try:
    import yfinance as yf
except ImportError:
    print("Installing yfinance... run: pip install yfinance")
    raise

# ML & preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Visualization
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OIL_TICKER = "CL=F"          # WTI Crude Oil futures (Yahoo Finance)
LOOKBACK_DAYS = 30           # Days of lagged features
TEST_SIZE = 0.2              # 20% for testing (last part of timeline)
RANDOM_STATE = 42
OUTPUT_DIR = "output"        # Folder for saving plots

# ---------------------------------------------------------------------------
# Fetch historical oil price data from Yahoo Finance
# ticker: the ticker symbol of the oil price data
# years: the number of years of data to fetch
# returns: a pandas DataFrame containing the oil price data
# ---------------------------------------------------------------------------
def fetch_oil_data(ticker=OIL_TICKER, years=5):
    end = datetime.now()
    start = end - timedelta(days=years * 365)
    print(f"Fetching data for {ticker} from {start.date()} to {end.date()}...")
    data = yf.download(ticker, start=start, end=end, progress=False)
    if data.empty or len(data) < 100:
        raise ValueError("Insufficient data. Check ticker or date range.")
    # Drop completely empty rows
    data = data.dropna(how="all")
    # Flatten MultiIndex columns (yfinance sometimes returns them)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]
    # Convert column names to lowercase and replace spaces with underscores
    data.columns = [str(c).replace(" ", "_").lower() for c in data.columns]
    print(f"Downloaded {len(data)} records.")
    return data


# ---------------------------------------------------------------------------
# Create lagged and rolling features from OHLCV for prediction.
# df: the pandas DataFrame containing the oil price data
# lookback: the number of days of lagged features to create
# returns: a pandas DataFrame containing the features
# ---------------------------------------------------------------------------
def create_features(df, lookback=LOOKBACK_DAYS):
    # Create a copy of the DataFrame to avoid modifying the original data
    df = df.copy()
    # Get the closing price column
    # If the close column is not present, use the adjusted close column
    close = df["close"] if "close" in df.columns else df["adj_close"]

    # Lagged closing prices
    # Create a new column for each lag (1, 5, 10)
    for lag in [1, 5, 10]:
        df[f"close_lag_{lag}"] = close.shift(lag)

    # Rolling statistics
    df["sma_5"] = close.rolling(5).mean()           # short-term moving average (5 days)
    df["sma_20"] = close.rolling(20).mean()         # long-term moving average (20 days)
    df["volatility_10"] = close.rolling(10).std()   # volatility (10 days)
    if "high" in df.columns and "low" in df.columns:
        df["range_5"] = (df["high"] - df["low"]).rolling(5).mean() # average of (High − Low) over 5 days
    df["target"] = close.shift(-1)                  # target is the next day's closing price (shift the closing price by 1 day)

    df = df.dropna()
    return df

# ---------------------------------------------------------------------------
# Get the column names used as model features (excluding target and date).
# returns: a list of column names
# ---------------------------------------------------------------------------
def get_feature_columns():
    return [
        "close_lag_1", "close_lag_5", "close_lag_10",
        "sma_5", "sma_20", "volatility_10", "range_5"
    ]

# ---------------------------------------------------------------------------
# Split into feature matrix X and target y; return index for plotting.
# df: the pandas DataFrame containing the oil price data
# feature_cols: the list of column names to use as features
# returns: a tuple containing the feature matrix X, the target y, and the index
# ---------------------------------------------------------------------------
def prepare_xy(df, feature_cols):
    # Create a copy of the DataFrame to avoid modifying the original data
    X = df[feature_cols].copy()
    # Drop rows with any remaining NaN
    # This is to ensure that the data is valid for training the model
    valid = X.notna().all(axis=1)

    X = X.loc[valid].astype(float)      # features
    y = df.loc[valid, "target"]         # target
    idx = df.index[valid]               # index
    return X, y, idx

# ---------------------------------------------------------------------------
# Train Random Forest, SVR, and MLP; return models, scaler, and metrics.
# X_train: the feature matrix for the training data
# X_test: the feature matrix for the testing data
# y_train: the target for the training data
# y_test: the target for the testing data
# scaler: the scaler to use for the data
# returns: a tuple containing the models, the scaler, and the metrics
# ---------------------------------------------------------------------------
def train_and_evaluate(X_train, X_test, y_train, y_test, scaler=None):
    # If the scaler is not provided, create a new one
    if scaler is None:
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
    else:
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
        "Support Vector Regression": SVR(kernel="rbf", C=10.0, gamma="scale"),
        "Neural Network": MLPRegressor(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            random_state=RANDOM_STATE,
            early_stopping=True,
        ),
    }
    
    # Train the models and evaluate the performance
    results = {}
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)
        results[name] = {
            "model": model,
            "predictions": pred,
            "mae": mean_absolute_error(y_test, pred),
            "rmse": np.sqrt(mean_squared_error(y_test, pred)),
            "r2": r2_score(y_test, pred),
        }
        print(f"  {name}: MAE={results[name]['mae']:.2f}, RMSE={results[name]['rmse']:.2f}, R²={results[name]['r2']:.4f}")

    return results, scaler

# ---------------------------------------------------------------------------
# Plot actual vs predicted prices for each model.
# y_test: the actual target values
# predictions_dict: a dictionary containing the predictions for each model
# dates_test: the dates for the testing data
# save_dir: the directory to save the plots
# returns: None
# ---------------------------------------------------------------------------
def plot_actual_vs_predicted(y_test, predictions_dict, dates_test, save_dir=OUTPUT_DIR):
    os.makedirs(save_dir, exist_ok=True)
    n_models = len(predictions_dict)
    fig, axes = plt.subplots(n_models, 1, figsize=(10, 3 * n_models), sharex=True)
    if n_models == 1:
        axes = [axes]
    dates_plot = range(len(dates_test))

    for ax, (name, pred) in zip(axes, predictions_dict.items()):
        ax.plot(dates_plot, y_test.values, label="Actual", color="steelblue", alpha=0.8)
        ax.plot(dates_plot, pred, label="Predicted", color="coral", alpha=0.8)
        ax.set_ylabel("Price (USD)")
        ax.set_title(f"Actual vs Predicted - {name}")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Test sample index")
    plt.tight_layout()
    path = os.path.join(save_dir, "actual_vs_predicted.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

# ---------------------------------------------------------------------------
# Use the most recent data to predict next day close; print result.
# df: the pandas DataFrame containing the oil price data
# feature_cols: the list of column names to use as features
# results: a dictionary containing the results for each model
# scaler: the scaler to use for the data
# returns: None
# ---------------------------------------------------------------------------
def print_next_day_prediction(df, feature_cols, results, scaler):
    # Get the last row of the DataFrame
    last = df[feature_cols].iloc[-1:]
    if last.isna().any().any():
        print("Skipping next-day prediction (missing values in last row).")
        return
    X_last = scaler.transform(last)
    print("\n--- Next trading day closing price prediction (most recent model: Random Forest) ---")
    rf = results["Random Forest"]["model"]
    pred = rf.predict(X_last)[0]
    print(f"  Predicted next close: ${pred:.2f} USD")
    print("---\n")

# ---------------------------------------------------------------------------
# Main function - run the oil price prediction app.
# returns: None
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Oil Price Prediction App")
    print("=" * 60)

    # 1. Load data
    df = fetch_oil_data(OIL_TICKER, years=5)
    feature_cols = get_feature_columns()

    # 2. Feature engineering
    df = create_features(df, lookback=LOOKBACK_DAYS)
    # Ensure we only use columns that exist (e.g. range_5 may be NaN if no high/low)
    available = [c for c in feature_cols if c in df.columns and df[c].notna().all()]
    if len(available) < len(feature_cols):
        feature_cols = available
    X, y, idx = prepare_xy(df, feature_cols)

    # 3. Time-based train/test split (last TEST_SIZE for test)
    n = len(X)
    split_idx = int(n * (1 - TEST_SIZE))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    dates_test = idx[split_idx:]

    print(f"\nTrain samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Features: {feature_cols}")

    # 4. Train and evaluate models
    print("\nModel performance (test set):")
    results, scaler = train_and_evaluate(X_train, X_test, y_train, y_test)

    # 5. Best model summary
    best_name = max(results, key=lambda k: results[k]["r2"])
    print(f"\nBest model by R²: {best_name} (R² = {results[best_name]['r2']:.4f})")

    # 6. Actual vs predicted plot (for Results section)
    print("\nGenerating plot...")
    plot_actual_vs_predicted(
        y_test,
        {k: v["predictions"] for k, v in results.items()},
        dates_test,
    )

    # 7. Next-day prediction
    print_next_day_prediction(df, feature_cols, results, scaler)

    print("Done. Check the 'output' folder for the prediction plot.")
    print("=" * 60)


if __name__ == "__main__":
    main()
