import os
import requests
import pandas as pd
import numpy as np

CITIES = ["London", "Manchester", "Birmingham", "Glasgow", "Edinburgh"]

METOFFICE_OBS_KEY = os.getenv("METOFFICE_OBSERVATION_API_KEY")
METOFFICE_MODEL_KEY = os.getenv("METOFFICE_MODEL_API_KEY")

BASE_OBS_URL = "https://api-metoffice.cda.climatecloud.com/observation-land/1"
BASE_MODEL_URL = "https://api-metoffice.cda.climatecloud.com/atmospheric-models/1.0.0"

def get_land_observation(city):
    # nearest endpoint
    resp = requests.get(
        f"{BASE_OBS_URL}/nearest",
        params={"lat": 51.5072, "lon": -0.1276},  # örnek London koordinat
        headers={"Authorization": f"Bearer {METOFFICE_OBS_KEY}"}
    )
    if resp.status_code != 200:
        return None
    data = resp.json()
    geohash = data["geohash"]

    # observations for geohash
    resp2 = requests.get(
        f"{BASE_OBS_URL}/{geohash}",
        headers={"Authorization": f"Bearer {METOFFICE_OBS_KEY}"}
    )
    if resp2.status_code != 200:
        return None
    obs = resp2.json()
    df = pd.DataFrame(obs["observations"])
    return df

def get_model_predictions(city):
    # GET orders
    orders_resp = requests.get(
        f"{BASE_MODEL_URL}/orders",
        headers={"Authorization": f"Bearer {METOFFICE_MODEL_KEY}"}
    )
    if orders_resp.status_code != 200:
        return None
    orders = orders_resp.json()
    if len(orders) == 0:
        return None

    order_id = orders[0]["orderId"]  # en son order
    files_resp = requests.get(
        f"{BASE_MODEL_URL}/orders/{order_id}/latest",
        headers={"Authorization": f"Bearer {METOFFICE_MODEL_KEY}"}
    )
    files = files_resp.json()
    file_id = files[0]["fileId"]

    # get data
    data_resp = requests.get(
        f"{BASE_MODEL_URL}/orders/{order_id}/latest/{file_id}/data",
        headers={"Authorization": f"Bearer {METOFFICE_MODEL_KEY}"},
        params={"dataSpec": "1.1.0"}
    )
    if data_resp.status_code != 200:
        return None
    # burayı kendi GRIB/JSON işleme mantığınla parse et
    return data_resp.json()

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
