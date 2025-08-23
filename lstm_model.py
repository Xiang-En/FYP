# lstm_model.py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# --------------------
# Config
# --------------------
SEQ_LEN = 60
MODEL_PATH = "models/lstm_model.keras"
SCALER_PATH = "models/lstm_scaler.pkl"

# --------------------
# Helpers
# --------------------
def _ensure_flat_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def _build_model(input_timesteps: int, input_features: int) -> Sequential:
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(input_timesteps, input_features)),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def _make_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["SMA5"] = out["Close"].rolling(5).mean()
    out["SMA10"] = out["Close"].rolling(10).mean()
    out["SMA20"] = out["Close"].rolling(20).mean()
    out.dropna(inplace=True)
    return out

def _prepare_data(df: pd.DataFrame, seq_len: int = SEQ_LEN) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler, int]:
    df = _ensure_flat_ohlc(df)
    if "Close" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'Close' column.")

    df = _make_indicators(df)

    feature_cols = ["Close", "SMA5", "SMA10", "SMA20"]
    data_raw = df[feature_cols].astype(np.float32).values  # (T, n_features)
    closes = df["Close"].astype(np.float32).values

    if len(df) < seq_len + 2:
        raise ValueError(f"Not enough data to build sequences (need >= {seq_len + 2}, got {len(df)}).")

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data_raw)
    n_features = data_scaled.shape[1]

    X_list, y_list = [], []
    for i in range(len(df) - seq_len - 1):
        X_list.append(data_scaled[i : i + seq_len, :])
        up = 1 if closes[i + seq_len + 1] > closes[i + seq_len] else 0
        y_list.append(up)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32).reshape(-1, 1)

    return X, y, scaler, n_features

# --------------------
# Public API
# --------------------
def train_or_load_model(df: pd.DataFrame,
                        seq_len: int = SEQ_LEN,
                        model_path: str = MODEL_PATH,
                        scaler_path: str = SCALER_PATH):
    """
    Train (or load) an LSTM on multivariate features [Close, SMA5, SMA10, SMA20].
    Returns: model, scaler, X_test, y_test, test_accuracy
    """
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)

    # Build eval arrays from provided df
    X_all, y_all, scaler_new, n_features = _prepare_data(df, seq_len=seq_len)
    split = int(len(X_all) * 0.8)
    X_train, y_train = X_all[:split], y_all[:split]
    X_test,  y_test  = X_all[split:], y_all[split:]

    # Try load existing
    model_loaded, scaler = None, None
    if Path(model_path).exists() and Path(scaler_path).exists():
        try:
            model_loaded = load_model(model_path)
            scaler = joblib.load(scaler_path)
            in_shape = model_loaded.input_shape  # (None, seq_len, n_features)
            if in_shape[1] != seq_len or in_shape[2] != n_features:
                model_loaded = None
        except Exception:
            model_loaded = None

    if model_loaded is None:
        model = _build_model(input_timesteps=seq_len, input_features=n_features)
        es = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=60,
            batch_size=32,
            callbacks=[es],
            verbose=0
        )
        model.save(model_path)
        joblib.dump(scaler_new, scaler_path)
        scaler = scaler_new
    else:
        model = model_loaded

    # Evaluate (using default 0.5 just to report a number; app tunes its own threshold)
    y_prob = model.predict(X_test, verbose=0)
    y_pred = (y_prob >= 0.5).astype(np.int32)
    test_acc = float(accuracy_score(y_test.astype(np.int32), y_pred))

    return model, scaler, X_test, y_test, test_acc
