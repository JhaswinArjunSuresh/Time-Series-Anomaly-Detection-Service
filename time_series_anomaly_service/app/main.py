from fastapi import FastAPI
from .models import TimeSeriesData
from .prophet_detector import detect_with_prophet
from .lstm_detector import detect_with_lstm

app = FastAPI()

@app.post("/detect_prophet")
def detect_prophet(data: TimeSeriesData):
    series_data = [{"ds": p.timestamp, "y": p.value} for p in data.series]
    results = detect_with_prophet(series_data)
    return {"results": results}

@app.post("/detect_lstm")
def detect_lstm(data: TimeSeriesData):
    series_data = [{"timestamp": p.timestamp, "value": p.value} for p in data.series]
    results = detect_with_lstm(series_data)
    return {"results": results}

@app.get("/health")
def health_check():
    return {"status": "ok"}
