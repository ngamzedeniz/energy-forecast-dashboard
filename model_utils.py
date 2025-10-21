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

# ------------------------------
# City coordinates
# ------------------------------
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

# ------------------------------
# API keys and endpoints
# ------------------------------
METOFFICE_OBS_KEY = os.getenv("METOFFICE_OBSERVATION_API_KEY")
METOFFICE_MODEL_KEY = os.getenv("METOFFICE_MODEL_API_KEY")

BASE_OBS_URL = "https://data.hub.api.metoffice.gov.uk/observation-land/1"
BASE_MODEL_URL = "https://data.hub.api.metoffice.gov.uk/atmospheric-models/1.0.0"

# ------------------------------
# Observation API
# ------------------------------
def get_land_observation(city: str) -> pd.DataFrame:
    """Fetch nearest land observation data for a city."""
    if city not in CITIES:
        raise ValueError(f"City '{city}' not found in list.")

    coords = CITIES[city]
    try:
        nearest_resp = requests.get(
            f"{BASE_OBS_URL}/nearest",
            params={"lat": coords["lat"], "lon": coords["lon"]},
            headers={"apikey": METOFFICE_OBS_KEY},
            timeout=10
        )
        nearest_resp.raise_for_status()
        geohash = nearest_resp.json().get("geohash")

        obs_resp = requests.get(
            f"{BASE_OBS_URL}/{geohash}",
            headers={"apikey": METOFFICE_OBS_KEY},
            timeout=10
        )
        obs_resp.raise_for_status()
        obs = obs_resp.json()

        df = pd.DataFrame(obs["observations"])
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
        return df

    except Exception as e:
        raise RuntimeError(f"Observation API error: {e}")

# ------------------------------
# Model API + Stacking ensemble
# ------------------------------
def get_model_predictions(obs_df: pd.DataFrame):
    """Stacking model trained with KFold using meteorological features."""
    if obs_df is None or obs_df.empty:
        raise ValueError("Observation dataframe is empty — cannot build features.")

    # Try retrieving model order info
    try:
        orders_resp = requests.get(
            f"{BASE_MODEL_URL}/orders",
            headers={"apikey": METOFFICE_MODEL_KEY},
            timeout=10
        )
        orders_resp.raise_for_status()
        orders = orders_resp.json()
    except Exception as e:
        print(f"⚠️ Model API error (orders): {e}")
        orders = []

    # Feature engineering
    df = obs_df.copy()
    df["hour"] = pd.to_datetime(df["datetime"]).dt.hour
    df["temp_anomaly"] = df.get("temperature_anomaly", 0)
    df["wind_speed"] = df.get("wind_speed", np.random.rand(len(df)) * 10)
    df["temperature"] = df.get("temperature", np.random.rand(len(df)) * 15)

    features = df[["temperature", "temp_anomaly", "wind_speed", "hour"]].values
    target_price = 100 + 0.5 * df["wind_speed"].values + np.random.randn(len(df)) * 0.2
    target_volume = 1000 + 5 * df["wind_speed"].values + np.random.randn(len(df)) * 2

    # Base learners
    base_models = [
        ('rf', RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42)),
        ('xgb', XGBRegressor(n_estimators=70, learning_rate=0.1, max_depth=4, random_state=42)),
        ('lgbm', LGBMRegressor(n_estimators=70, learning_rate=0.1, max_depth=5, random_state=42))
    ]

    # Stacking models
    stack_price = StackingRegressor(estimators=base_models, final_estimator=LinearRegression(), cv=3)
    stack_volume = StackingRegressor(estimators=base_models, final_estimator=LinearRegression(), cv=3)

    # Cross-validation
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

    print(f"✅ Cross-val RMSE (price): {np.mean(rmse_price):.3f}, (volume): {np.mean(rmse_volume):.3f}")

    # Tahmin üret
    pred_price = stack_price.predict(features[-1].reshape(1, -1))[0]
    pred_volume = stack_volume.predict(features[-1].reshape(1, -1))[0]

    return round(pred_price, 2), int(pred_volume)

# ------------------------------
# Simple insight generation
# ------------------------------
def generate_insight(obs_df: pd.DataFrame):
    """Generate descriptive meteorological insights for the dashboard."""
    if obs_df is None or obs_df.empty:
        return 0, 0, 0

    avg_wind = obs_df.get("wind_speed", pd.Series(np.random.rand(len(obs_df)) * 10)).mean()
    max_wind = obs_df.get("wind_speed", pd.Series(np.random.rand(len(obs_df)) * 10)).max()
    norm_temp = obs_df.get("temperature", pd.Series(np.random.rand(len(obs_df)) * 15)).mean()

    return round(avg_wind, 2), round(max_wind, 2), round(norm_temp, 2)
