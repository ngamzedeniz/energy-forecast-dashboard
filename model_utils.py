# model_utils.py (Gerçek Veri, Stacking Model, 30 UK Şehri)
import pandas as pd
import numpy as np
import os
import requests
import json
import datetime
from urllib import parse
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# --- Şehir verilerini yükleme ---
with open("cities.json", "r") as f:
    CITIES_DATA = json.load(f)
    CITIES_DICT = {c['name']: c for c in CITIES_DATA}

# --- HDD/CDD Hesaplama ---
def calculate_hdd_cdd(temp_c: float, base_temp: float = 18.0, hdd: bool = True) -> float:
    return max(0, base_temp - temp_c) if hdd else max(0, temp_c - base_temp)

# --- OpenMeteo API ---
def get_openmeteo_data(lat: float, lon: float) -> pd.DataFrame:
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=5)
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": "temperature_2m,windspeed_10m",
        "timezone": "auto",
        "temperature_unit": "celsius",
        "wind_speed_unit": "ms"
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if 'hourly' not in data:
            return pd.DataFrame()
        hourly = data['hourly']
        df = pd.DataFrame({
            'Time': pd.to_datetime(hourly['time']),
            'Actual_Temp_C': hourly['temperature_2m'],
            'Actual_WindSpeed': hourly['windspeed_10m']
        })
        df.set_index('Time', inplace=True)
        df['API_Source'] = 'OpenMeteo'
        return df
    except Exception as e:
        print(f"OpenMeteo error: {e}")
        return pd.DataFrame()

# --- Met Office API ---
def get_metoffice_data(city: str) -> pd.DataFrame:
    city_data = CITIES_DICT.get(city)
    if not city_data:
        return pd.DataFrame()
    geohash = city_data.get('geohash')
    lat = city_data.get('lat')
    lon = city_data.get('lon')
    token = os.getenv('METOFFICE_TOKEN')
    if geohash and token:
        url = f"https://data.hub.api.metoffice.gov.uk/observation-land/1/{geohash}"
        try:
            resp = requests.get(url, params={"apikey": token}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if 'features' in data:
                records = []
                for f in data['features']:
                    rec = f['properties']
                    records.append({
                        'Time': pd.to_datetime(rec['datetime']).tz_localize('UTC'),
                        'Actual_Temp_C': rec.get('temperature'),
                        'Actual_WindSpeed': rec.get('wind_speed')
                    })
                df = pd.DataFrame(records)
                df.set_index('Time', inplace=True)
                df['API_Source'] = 'Met Office'
                return df
        except Exception as e:
            print(f"Met Office error: {e}")
    if lat and lon:
        return get_openmeteo_data(lat, lon)
    return pd.DataFrame()

# --- NESO API: UK Spot Price ---
def get_neso_price(city: str, limit: int = 200) -> pd.DataFrame:
    try:
        sql_query = f'SELECT * FROM "b2bde559-3455-4021-b179-dfe60c0337b0" WHERE city="{city}" ORDER BY "_id" ASC LIMIT {limit}'
        params = {'sql': sql_query}
        url = 'https://api.neso.energy/api/3/action/datastore_search_sql'
        response = requests.get(url, params=parse.urlencode(params), timeout=15)
        response.raise_for_status()
        data = response.json()["result"]["records"]
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df['Time'] = pd.to_datetime(df['timestamp'])
        df.set_index('Time', inplace=True)
        df = df[['price']].rename(columns={'price':'SpotPrice_EUR'})
        return df
    except Exception as e:
        print(f"NESO GET ERROR for {city}: {e}")
        return pd.DataFrame()

# --- Batch Forecast for 30 cities ---
def run_energy_forecast(city: str) -> pd.DataFrame:
    weather_df = get_metoffice_data(city)
    if weather_df.empty:
        print(f"No weather data for {city}")
        return pd.DataFrame()
    api_source = weather_df.get('API_Source', pd.Series(['Unknown'])).iloc[0]
    weather_df = weather_df.drop(columns=['API_Source'], errors='ignore')

    spot_df = get_neso_price(city)
    
    df_list = [weather_df]
    if not spot_df.empty:
        df_list.append(spot_df)
    merged_df = pd.concat(df_list, axis=1).sort_index()
    merged_df.fillna(method='ffill', inplace=True)

    # --- Stacking Model ---
    merged_df['Target_Actual'] = merged_df.get('SpotPrice_EUR', 0)  # as target
    merged_df['WindGen_MW'] = merged_df.get('Actual_WindSpeed', 0) * 10  # dummy wind effect

    feature_cols = ['Actual_Temp_C','Actual_WindSpeed','WindGen_MW']
    X = merged_df[feature_cols].fillna(0)
    y = merged_df['Target_Actual'].fillna(0)

    base_models = [
        ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
        ('xgb', XGBRegressor(n_estimators=50, random_state=42, verbosity=0))
    ]
    stack_model = StackingRegressor(estimators=base_models, final_estimator=RandomForestRegressor(n_estimators=50, random_state=42))
    if len(X) > 1:
        stack_model.fit(X, y)
        merged_df['Final_Stacking_Prediction'] = stack_model.predict(X)
    else:
        merged_df['Final_Stacking_Prediction'] = y  # fallback

    merged_df['API_Source'] = api_source
    return merged_df

# --- TEST / DEBUG ---
if __name__ == "__main__":
    print("⏳ 30 UK city batch forecast starting...")
    for city_entry in CITIES_DATA:
        city_name = city_entry['name']
        df = run_energy_forecast(city_name)
        if df.empty:
            print(f"{city_name}: Forecast FAILED")
        else:
            print(f"{city_name}: Forecast OK, rows={len(df)}")
