import os
import requests
import pandas as pd

# Şehirler ve koordinatları
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

# Environment variable’dan API key’ler
METOFFICE_OBS_KEY = os.getenv("METOFFICE_OBSERVATION_API_KEY")
METOFFICE_MODEL_KEY = os.getenv("METOFFICE_MODEL_API_KEY")

# Base URL’ler
BASE_OBS_URL = "https://data.hub.api.metoffice.gov.uk/observation-land/1"
BASE_MODEL_URL = "https://data.hub.api.metoffice.gov.uk/atmospheric-models/1.0.0"

def get_land_observation(city):
    if city not in CITIES:
        raise Exception(f"City '{city}' not found")
    lat = CITIES[city]["lat"]
    lon = CITIES[city]["lon"]

    # Nearest land observation
    resp = requests.get(
        f"{BASE_OBS_URL}/nearest",
        params={"lat": lat, "lon": lon},
        headers={"Authorization": f"Bearer {METOFFICE_OBS_KEY}"}
    )
    if resp.status_code != 200:
        raise Exception(f"Observation nearest API error: {resp.status_code} {resp.text}")
    geohash = resp.json()["geohash"]

    # Observations for geohash
    resp2 = requests.get(
        f"{BASE_OBS_URL}/{geohash}",
        headers={"Authorization": f"Bearer {METOFFICE_OBS_KEY}"}
    )
    if resp2.status_code != 200:
        raise Exception(f"Observation data API error: {resp2.status_code} {resp2.text}")

    obs = resp2.json()
    df = pd.DataFrame(obs["observations"])
    return df

def get_model_predictions(ticker):
    """
    Atmospheric model API’den gerçek tahminleri alır.
    Örnek: price ve volume bilgisi JSON’da dönüyor.
    Burada GRIB dosyalarını parse etmek gerekebilir.
    """
    # En son order
    orders_resp = requests.get(
        f"{BASE_MODEL_URL}/orders",
        headers={"Authorization": f"Bearer {METOFFICE_MODEL_KEY}"}
    )
    if orders_resp.status_code != 200:
        raise Exception(f"Model orders API error: {orders_resp.status_code} {orders_resp.text}")
    orders = orders_resp.json()
    if len(orders) == 0:
        raise Exception("No model orders found")

    order_id = orders[0]["orderId"]
    files_resp = requests.get(
        f"{BASE_MODEL_URL}/orders/{order_id}/latest",
        headers={"Authorization": f"Bearer {METOFFICE_MODEL_KEY}"}
    )
    files = files_resp.json()
    if len(files) == 0:
        raise Exception("No files in latest order")

    file_id = files[0]["fileId"]

    # GRIB/JSON veri
    data_resp = requests.get(
        f"{BASE_MODEL_URL}/orders/{order_id}/latest/{file_id}/data",
        headers={
            "Authorization": f"Bearer {METOFFICE_MODEL_KEY}",
            "Accept": "application/x-grib"
        },
        params={"dataSpec": "1.1.0"}
    )
    if data_resp.status_code not in [200, 302]:
        raise Exception(f"Model data API error: {data_resp.status_code} {data_resp.text}")

    # Burada GRIB parse edilip price/volume çıkarılmalı
    # Şu an örnek olarak JSON dönüyorsa:
    try:
        data_json = data_resp.json()
        predicted_price = data_json.get("price", 0)
        predicted_volume = data_json.get("volume", 0)
    except:
        predicted_price = 0
        predicted_volume = 0

    return predicted_price, predicted_volume

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
