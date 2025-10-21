import os
import pandas as pd
import numpy as np
import httpx
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor

# --- API KEYS ---
OBSERVATION_API_KEY = os.getenv("METOFFICE_OBSERVATION_API_KEY")
MODEL_API_KEY = os.getenv("METOFFICE_MODEL_API_KEY")

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
    "Inverness": {"lat": 57.4778, "lon": -4.2247}
}
CITIES = list(CITY_COORDINATES.keys())

# --- UK Monthly Norm Temp (°C) ---
UK_MONTHLY_NORM_TEMP = {
    1: 5.5, 2: 5.5, 3: 7.5, 4: 9.5, 5: 13.0, 6: 15.5,
    7: 17.5, 8: 17.5, 9: 15.0, 10: 12.0, 11: 8.5, 12: 6.5
}

# --- WEATHER DATA ---
async def get_weather_data(city_name: str, hours: int = 48):
    if city_name not in CITY_COORDINATES:
        return None, f"Invalid city: {city_name}"
    
    if not OBSERVATION_API_KEY:
        return None, "Missing Met Office Observation API key"
    
    coords = CITY_COORDINATES[city_name]
    url = f"https://api.metoffice.gov.uk/land-observations/nearest?lat={coords['lat']}&lon={coords['lon']}"
    headers = {"Authorization": f"Bearer {OBSERVATION_API_KEY}"}
    
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.get(url, headers=headers)
            if resp.status_code != 200:
                return None, f"Observation API error: {resp.status_code}"
            data = resp.json().get("features", [])
            if not data:
                return None, "No observation data returned"

            df = pd.DataFrame([{
                "Time": datetime.fromisoformat(f["properties"]["Date"][:-1]),
                "Temperature": f["properties"].get("air_temperature"),
                "Wind_Speed": f["properties"].get("wind_speed"),
                "Wind_Direction": f["properties"].get("wind_direction"),
                "Cloud_Cover": f["properties"].get("total_cloud_cover"),
                "Precipitation": f["properties"].get("precipitation_amount", 0)
            } for f in data[:hours]])

            month = df["Time"].iloc[0].month
            df["Temp_Anomaly"] = df["Temperature"] - UK_MONTHLY_NORM_TEMP.get(month, 10.0)
            return df, None
        except Exception as e:
            return None, str(e)

# --- MODEL PREDICTIONS ---
async def get_model_predictions(df_weather, target_col="Wind_Speed", feature_cols=None):
    if feature_cols is None:
        feature_cols = ["Temperature", "Wind_Speed", "Cloud_Cover", "Temp_Anomaly"]

    # Stacking model
    X = df_weather[feature_cols].values
    y = df_weather[target_col].values

    estimators = [
        ('lr', LinearRegression()),
        ('dt', DecisionTreeRegressor(max_depth=5)),
        ('rf', RandomForestRegressor(n_estimators=50, max_depth=5))
    ]
    stack_model = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
    stack_model.fit(X, y)
    preds = stack_model.predict(X)
    return preds

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
