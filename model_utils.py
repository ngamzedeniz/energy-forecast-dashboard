import os
import requests
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Cities ve koordinatlar
CITIES = {
    "London": {"lat": 51.5074, "lon": 0.1278},
    "Manchester": {"lat": 53.4808, "lon": 2.2426},
    "Birmingham": {"lat": 52.4862, "lon": 1.8904},
    "Leeds": {"lat": 53.8008, "lon": 1.5491},
    "Glasgow": {"lat": 55.8642, "lon": 4.2518},
    "Liverpool": {"lat": 53.4084, "lon": 2.9916},
    "Newcastle": {"lat": 54.9783, "lon": 1.6178},
    "Sheffield": {"lat": 53.3829, "lon": 1.4659},
    "Bristol": {"lat": 51.4545, "lon": 2.5879},
    "Edinburgh": {"lat": 55.9533, "lon": 3.1883}
}

# API key'ler
METOFFICE_OBS_KEY = os.getenv("METOFFICE_OBSERVATION_API_KEY")
METOFFICE_MODEL_KEY = os.getenv("METOFFICE_MODEL_API_KEY")

# Base URL'ler
BASE_OBS_URL = "https://data.hub.api.metoffice.gov.uk/observation-land/1"
BASE_MODEL_URL = "https://data.hub.api.metoffice.gov.uk/atmospheric-models/1.0.0"

# ------------------------------
# API çağrıları
# ------------------------------
def get_land_observation(city):
    coords = CITIES[city]
    try:
        # nearest
        resp = requests.get(
            f"{BASE_OBS_URL}/nearest",
            params={"lat": coords["lat"], "lon": coords["lon"]},
            headers={"apikey": METOFFICE_OBS_KEY}
        )
        resp.raise_for_status()
        data = resp.json()
        geohash = data["geohash"]

        # observations for geohash
        resp2 = requests.get(
            f"{BASE_OBS_URL}/{geohash}",
            headers={"apikey": METOFFICE_OBS_KEY}
        )
        resp2.raise_for_status()
        obs = resp2.json()
        df = pd.DataFrame(obs["observations"])
        return df
    except Exception as e:
        raise RuntimeError(f"Observation nearest API error: {e}")

def get_model_predictions(obs_df):
    """
    Stacking model ile predicted_price ve predicted_volume tahmini yapar.
    Örnek: Basit model, XGBoost, LightGBM, RandomForest stacking.
    """
    if obs_df is None or obs_df.empty:
        return 100.0, 1000  # fallback değer
    
    # Feature engineering
    df = obs_df.copy()
    df["hour"] = pd.to_datetime(df["datetime"]).dt.hour
    df["temp_anomaly"] = df.get("temperature_anomaly", 0)
    
    features = df[["temperature", "temp_anomaly", "wind_speed", "hour"]].values
    target_price = 100 + df["wind_speed"].values  # örnek ilişki, gerçek model için değiştirilebilir
    target_volume = 1000 + df["wind_speed"].values  # örnek

    # Stacking setup
    base_models = [
        ('rf', RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)),
        ('xgb', XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)),
        ('lgbm', LGBMRegressor(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=42))
    ]
    stack_price = StackingRegressor(estimators=base_models, final_estimator=LinearRegression(), cv=3)
    stack_volume = StackingRegressor(estimators=base_models, final_estimator=LinearRegression(), cv=3)

    # CV ile fit
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    rmse_price, rmse_volume = [], []

    for tr_idx, val_idx in kf.split(features):
        X_tr, X_val = features[tr_idx], features[val_idx]
        y_tr_price, y_val_price = target_price[tr_idx], target_price[val_idx]
        y_tr_vol, y_val_vol = target_volume[tr_idx], target_volume[val_idx]

        stack_price.fit(X_tr, y_tr_price)
        stack_volume.fit(X_tr, y_tr_vol)

        rmse_price.append(np.sqrt(mean_squared_error(y_val_price, stack_price.predict(X_val))))
        rmse_volume.append(np.sqrt(mean_squared_error(y_val_vol, stack_volume.predict(X_val))))

    # Tahmin
    pred_price = stack_price.predict(features[-1].reshape(1, -1))[0]
    pred_volume = stack_volume.predict(features[-1].reshape(1, -1))[0]

    return round(pred_price, 2), int(pred_volume)

def generate_insight(obs_df):
    if obs_df is None or obs_df.empty:
        return {"avg_wind": 0, "max_wind": 0, "norm_temp": 0, "table": []}
    avg_wind = obs_df["wind_speed"].mean()
    max_wind = obs_df["wind_speed"].max()
    norm_temp = obs_df["temperature"].mean()
    table_data = []
    for idx, row in obs_df.iterrows():
        table_data.append([
            pd.to_datetime(row["datetime"]),
            row["wind_speed"],
            row["wind_direction"],
            row["temperature"],
            row.get("temperature_anomaly", 0),
            row.get("precipitation", 0),
            row.get("cloud_cover", 0)
        ])
    return {
        "avg_wind": round(avg_wind, 1),
        "max_wind": round(max_wind, 1),
        "norm_temp": round(norm_temp, 1),
        "table": table_data
    }
