import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def prepare_lstm_data(values, window=5):
    X, y = [], []
    for i in range(len(values) - window):
        X.append(values[i:i+window])
        y.append(values[i+window])
    return np.array(X), np.array(y)

def detect_with_lstm(series_data, window=5):
    values = [p['value'] for p in series_data]
    X, y = prepare_lstm_data(values, window)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(32, input_shape=(window,1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, verbose=0)

    preds = model.predict(X, verbose=0).flatten()
    residuals = np.abs(preds - y)
    threshold = np.mean(residuals) + 2*np.std(residuals)
    anomalies = residuals > threshold

    results = []
    for i, point in enumerate(series_data[window:]):
        results.append({
            "timestamp": point['timestamp'],
            "value": point['value'],
            "predicted": float(preds[i]),
            "anomaly": bool(anomalies[i])
        })
    return results
