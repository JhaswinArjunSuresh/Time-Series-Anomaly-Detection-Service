from pydantic import BaseModel
from typing import List

class TimeSeriesPoint(BaseModel):
    timestamp: str
    value: float

class TimeSeriesData(BaseModel):
    series: List[TimeSeriesPoint]
