import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import requests
import os

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
ELEXON_API_KEY = os.getenv("ELEXON_API_KEY")

def forecast_energy(ticker, df_weather):
    """
    Stacking model ile fiyat ve hacim tahmini.
    """
    end_date = pd.Timestamp.today().date()
    start_date = end_date - pd.Timedelta(days=365)
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        return None, None

    data['MA_7'] = data['Close'].rolling(7).mean()
    data['Vol_MA_7'] = data['Volume'].rolling(7).mean()
    data.dropna(inplace=True)

    X_price = data[['MA_7', 'Vol_MA_7']].values
    y_price = data['Close'].values

    X_volume = data[['Vol_MA_7', 'MA_7']].values
    y_volume = data['Volume'].values

    base_learners = [
        ('lr', LinearRegression()),
        ('dt', DecisionTreeRegressor())
    ]
    price_model = StackingRegressor(estimators=base_learners, final_estimator=LinearRegression())
    volume_model = StackingRegressor(estimators=base_learners, final_estimator=LinearRegression())

    price_model.fit(X_price, y_price)
    volume_model.fit(X_volume, y_volume)

    last_row_price = np.array([[data['MA_7'].iloc[-1], data['Vol_MA_7'].iloc[-1]]])
    last_row_vol = np.array([[data['Vol_MA_7'].iloc[-1], data['MA_7'].iloc[-1]]])

    future_price = price_model.predict(last_row_price).item()
    future_volume = volume_model.predict(last_row_vol).item()

    return future_price, future_volume


def get_weather_insight(city_coords):
    """
    OpenWeather API ile 48 saatlik hava tahmini ve meteorolojik insight.
    """
    lat, lon = city_coords["lat"], city_coords["lon"]
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
    try:
        r = requests.get(url)
        r.raise_for_status()
        data = r.json()
        df_weather = pd.DataFrame({
            "Time": [pd.to_datetime(item['dt_txt']) for item in data['list'][:16]],
            "Temp": [item['main']['temp'] for item in data['list'][:16]],
            "Wind": [item['wind']['speed'] for item in data['list'][:16]],
            "WindDir": [item['wind']['deg'] for item in data['list'][:16]],
            "Clouds": [item['clouds']['all'] for item in data['list'][:16]],
            "Precip": [item.get('rain', {}).get('3h',0)+item.get('snow', {}).get('3h',0) for item in data['list'][:16]]
        })
        avg_wind = df_weather["Wind"].mean()
        max_wind = df_weather["Wind"].max()
        temp_anomaly = df_weather["Temp"].mean() - 15  # Örnek norm
        if max_wind>7.5:
            interpretation = "High wind alert: Potentially good for wind energy."
        elif temp_anomaly>5:
            interpretation = "Heatwave anomaly: Peak cooling demand expected."
        elif temp_anomaly<-3:
            interpretation = "Cold spell anomaly: Peak heating demand expected."
        else:
            interpretation = "Stable weather conditions."
        return df_weather, avg_wind, max_wind, interpretation
    except:
        return None, None, None, "Weather data unavailable."
