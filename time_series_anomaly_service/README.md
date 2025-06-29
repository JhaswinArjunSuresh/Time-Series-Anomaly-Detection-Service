# 📈 Time-Series Anomaly Detection Service

This service detects anomalies in time-series data using:
- 📊 Prophet (trend + seasonality with prediction intervals)
- 🧠 Custom LSTM for residual-based anomaly detection

## Features
- Submit JSON time series (timestamps + values)
- Get flagged anomalies using either model
- FastAPI for simple REST usage

## Run locally
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
