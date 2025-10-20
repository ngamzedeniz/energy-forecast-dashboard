import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

WEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
ELEXON_API_KEY = os.getenv("ELEXON_API_KEY")

# --- UK & Scotland Cities (Wind & Solar Energy Relevance) ---
CITY_COORDINATES = {
    "London": {"lat": 51.5074, "lon": -0.1278},
    "Manchester": {"lat": 53.4808, "lon": -2.2426},
    "Birmingham": {"lat": 52.4862, "lon": -1.8904},
    "Leeds": {"lat": 53.8008, "lon": -1.5491},
    "Glasgow": {"lat": 55.8642, "lon": -4.2518},
    "Edinburgh": {"lat": 55.9533, "lon": -3.1883},
    "Aberdeen": {"lat": 57.1497, "lon": -2.0943},
    "Dundee": {"lat": 56.4620, "lon": -2.9707},
    "Inverness": {"lat": 57.4778, "lon": -4.2247},
    "Newcastle": {"lat": 54.9783, "lon": -1.6178},
    "Liverpool": {"lat": 53.4084, "lon": -2.9916},
    "Sheffield": {"lat": 53.3829, "lon": -1.4659},
    "Bristol": {"lat": 51.4545, "lon": -2.5879}
}
CITIES = list(CITY_COORDINATES.keys())

# --- UK Monthly Norm Temp (°C) ---
UK_MONTHLY_NORM_TEMP = {
    1: 5.5, 2: 5.5, 3: 7.5, 4: 9.5, 5: 13.0, 6: 15.5,
    7: 17.5, 8: 17.5, 9: 15.0, 10: 12.0, 11: 8.5, 12: 6.5
}

# --- WEATHER DATA ---
def get_weather_data(city_name: str, hours: int = 48):
    """Fetch next 48h forecast from OpenWeatherMap."""
    if city_name not in CITY_COORDINATES:
        return None, f"Invalid city: {city_name}"
    
    coords = CITY_COORDINATES[city_name]
    lat, lon = coords["lat"], coords["lon"]
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&units=metric&appid={WEATHER_API_KEY}"
    
    try:
        resp = requests.get(url)
        if resp.status_code != 200:
            return None, f"Weather API error: {resp.status_code}"
        data = resp.json()["list"][:hours//3]  # 3-hourly forecast
        df = pd.DataFrame({
            "Time": [datetime.fromisoformat(item["dt_txt"]) for item in data],
            "Temperature": [item["main"]["temp"] for item in data],
            "Wind_Speed": [item["wind"]["speed"] for item in data],
            "Wind_Direction": [item["wind"]["deg"] for item in data],
            "Cloud_Cover": [item["clouds"]["all"] for item in data],
            "Precipitation": [item.get("rain", {}).get("3h", 0) + item.get("snow", {}).get("3h", 0) for item in data]
        })
        month = df["Time"].iloc[0].month
        norm_temp = UK_MONTHLY_NORM_TEMP.get(month, 10.0)
        df["Temp_Anomaly"] = df["Temperature"] - norm_temp
        return df, None
    except Exception as e:
        return None, str(e)

# --- ELEXON DATA ---
def get_elexon_data(ticker: str, days: int = 365):
    """Fetch UK electricity data from Elexon API."""
    end = datetime.today()
    start = end - timedelta(days=days)
    url = f"https://api.bmreports.com/BMRS/{ticker}/v1?APIKey={ELEXON_API_KEY}&FromDate={start.strftime('%Y-%m-%d')}&ToDate={end.strftime('%Y-%m-%d')}"
    try:
        resp = requests.get(url)
        if resp.status_code != 200:
            return None, f"Elexon API error: {resp.status_code}"
        # Simplified: convert JSON/CSV response to DataFrame
        data = resp.json()["Response"]
        df = pd.DataFrame(data)
        return df, None
    except Exception as e:
        return None, str(e)

# --- STACKING MODEL ---
def train_stacking_model(df, target_col: str, feature_cols: list):
    """Train a stacking model for prediction."""
    X = df[feature_cols].values
    y = df[target_col].values
    estimators = [
        ('lr', LinearRegression()),
        ('dt', DecisionTreeRegressor(max_depth=5)),
        ('rf', RandomForestRegressor(n_estimators=50, max_depth=5))
    ]
    stack_model = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
    stack_model.fit(X, y)
    return stack_model

def predict_with_stacking(model, df, feature_cols: list):
    X = df[feature_cols].values
    return model.predict(X)

# --- METEOROLOGICAL INSIGHT ---
def get_meteorological_insight(df_weather):
    avg_wind = df_weather["Wind_Speed"].mean()
    max_wind = df_weather["Wind_Speed"].max()
    max_wind_time = df_weather["Time"].iloc[df_weather["Wind_Speed"].idxmax()]
    temp_anomaly_max = df_weather["Temp_Anomaly"].max()
    temp_anomaly_min = df_weather["Temp_Anomaly"].min()
    
    if max_wind > 7.5:
        interpretation = "Strong winds forecasted: positive impact on wind energy."
    elif temp_anomaly_max > 5:
        interpretation = "Heatwave anomaly: expect high cooling demand."
    elif temp_anomaly_min < -3:
        interpretation = "Cold spell anomaly: expect high heating demand."
    else:
        interpretation = "Stable weather: moderate market impact."
    
    return {
        "avg_wind": avg_wind,
        "max_wind": max_wind,
        "max_wind_time": max_wind_time,
        "interpretation": interpretation
    }
