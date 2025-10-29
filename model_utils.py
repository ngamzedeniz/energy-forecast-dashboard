# model_utils.py
import pandas as pd
import numpy as np
import requests
import json
import datetime
import yfinance as yf
from urllib.parse import urlencode
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression

# cities.json yükle
with open("cities.json", "r") as f:
    CITIES = json.load(f)

# NESO SQL → enerji fiyat verisi
def get_neso_price_data(limit=5000):
    try:
        sql_query = f'''SELECT * FROM "b2bde559-3455-4021-b179-dfe60c0337b0" ORDER BY "_id" ASC LIMIT {limit}'''
        params = {'sql': sql_query}
        url = "https://api.neso.energy/api/3/action/datastore_search_sql"
        r = requests.get(url, params=urlencode(params), timeout=20)
        j = r.json()
        records = j["result"]["records"]

        df = pd.DataFrame(records)
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)
        df.rename(columns={"price": "SpotPrice_EUR"}, inplace=True)
        df["SpotPrice_EUR"] = pd.to_numeric(df["SpotPrice_EUR"], errors="coerce")

        return df[["SpotPrice_EUR"]].dropna()

    except Exception as e:
        print("NESO GET ERROR:", e)
        return pd.DataFrame()

# Open-Meteo → hava
def get_weather(lat, lon):
    try:
        end = datetime.date.today()
        start = end - datetime.timedelta(days=5)
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "hourly": "temperature_2m,windspeed_10m",
            "timezone": "auto",
        }
        r = requests.get(url, params=params, timeout=20)
        j = r.json()
        hourly = j["hourly"]

        df = pd.DataFrame({
            "Time": pd.to_datetime(hourly["time"]),
            "Actual_Temp_C": hourly["temperature_2m"],
            "Actual_WindSpeed": hourly["windspeed_10m"]
        })
        df.set_index("Time", inplace=True)
        return df
    except:
        return pd.DataFrame()

# Spot fiyat yfinance (yedek)
def get_spot_backup():
    try:
        d = yf.download("EXC.L", period="5d", interval="1h")
        if d.empty: return pd.DataFrame()
        return pd.DataFrame({"SpotPrice_EUR": d["Close"]})
    except:
        return pd.DataFrame()

# --------- BATCH DATASET OLUŞTURMA ---------
def build_batch_dataset():
    print("⏳ 30 şehir için veri indiriliyor...")

    price_df = get_neso_price_data()
    if price_df.empty:
        price_df = get_spot_backup()

    all_data = []

    for city in CITIES:
        lat = city["lat"]
        lon = city["lon"]
        name = city["name"]

        weather_df = get_weather(lat, lon)
        if weather_df.empty: 
            continue

        merged = price_df.join(weather_df, how="inner")
        merged["City"] = name
        all_data.append(merged)

    if not all_data:
        return pd.DataFrame()

    big_df = pd.concat(all_data)
    big_df.sort_index(inplace=True)
    big_df.dropna(inplace=True)

    return big_df


# -------- STAKING MODEL EĞİTİMİ --------
def train_stacking_model(df):
    df = df.copy()
    df["Hour"] = df.index.hour
    df["Day"]  = df.index.day
    df["Month"] = df.index.month

    features = ["Actual_Temp_C", "Actual_WindSpeed", "Hour", "Day", "Month"]
    target = "SpotPrice_EUR"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2)

    model_rf = RandomForestRegressor(n_estimators=100)
    model_rf.fit(X_train, y_train)

    model_xgb = XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9
    )
    model_xgb.fit(X_train, y_train)

    df_stack = pd.DataFrame({
        "RF": model_rf.predict(X),
        "XGB": model_xgb.predict(X)
    }, index=df.index)

    meta = LinearRegression()
    meta.fit(df_stack, y)

    return model_rf, model_xgb, meta


# -------- TAHMİN FONKSİYONU --------
def run_energy_forecast(city_name: str):
    # 1) dataset yoksa oluştur
    df = build_batch_dataset()
    if df.empty:
        return pd.DataFrame()

    # 2) stacking modeli eğit
    rf, xgb, meta = train_stacking_model(df)

    # 3) seçilen şehir için veri çek
    city = next(c for c in CITIES if c["name"] == city_name)
    wdf = get_weather(city["lat"], city["lon"])
    if wdf.empty:
        return pd.DataFrame()

    wdf["Hour"] = wdf.index.hour
    wdf["Day"]  = wdf.index.day
    wdf["Month"] = wdf.index.month

    feats = ["Actual_Temp_C", "Actual_WindSpeed", "Hour", "Day", "Month"]
    wdf = wdf.dropna(subset=feats)

    # stacking tahmini
    rf_pred = rf.predict(wdf[feats])
    xgb_pred = xgb.predict(wdf[feats])
    stack_input = pd.DataFrame({"RF": rf_pred, "XGB": xgb_pred}, index=wdf.index)
    final_pred = meta.predict(stack_input)

    wdf["Final_Stacking_Prediction"] = final_pred
    wdf["API_Source"] = "NESO + OpenMeteo"

    return wdf
