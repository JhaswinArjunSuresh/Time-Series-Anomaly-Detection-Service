import pandas as pd
from prophet import Prophet

def detect_with_prophet(series_data):
    df = pd.DataFrame(series_data)
    df.columns = ['ds', 'y']
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=0)
    forecast = model.predict(future)
    df['yhat'] = forecast['yhat']
    df['yhat_lower'] = forecast['yhat_lower']
    df['yhat_upper'] = forecast['yhat_upper']
    df['anomaly'] = (df['y'] < df['yhat_lower']) | (df['y'] > df['yhat_upper'])
    return df.to_dict(orient='records')
