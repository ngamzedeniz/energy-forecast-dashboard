# model_utils.py (Güncellenmiş, Gerçek Veri ile)
import pandas as pd
import numpy as np
import os
import requests
import json
import datetime
import yfinance as yf
from requests.exceptions import RequestException

# --- Şehir Verilerini Yükleme ---
try:
    with open("cities.json", "r") as f:
        CITIES_DATA = json.load(f)
        CITIES_DICT = {c['name']: c for c in CITIES_DATA}
except FileNotFoundError:
    CITIES_DICT = {}

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
        if resp.status_code != 200: return pd.DataFrame()
        data = resp.json()
        if 'hourly' not in data: return pd.DataFrame()
        hourly = data['hourly']
        df = pd.DataFrame({
            'Time': pd.to_datetime(hourly['time']),
            'Actual_Temp_C': hourly['temperature_2m'],
            'Actual_WindSpeed': hourly['windspeed_10m']
        })
        df.set_index('Time', inplace=True)
        df['API_Source'] = 'OpenMeteo'
        return df
    except Exception:
        return pd.DataFrame()

# --- Met Office API ---
def get_metoffice_data(city: str) -> pd.DataFrame:
    city_data = CITIES_DICT.get(city)
    if not city_data: return pd.DataFrame()
    geohash = city_data.get('geohash')
    lat = city_data.get('lat')
    lon = city_data.get('lon')
    token = os.getenv('METOFFICE_TOKEN')
    if geohash and token:
        url = f"https://data.hub.api.metoffice.gov.uk/observation-land/1/{geohash}"
        try:
            resp = requests.get(url, params={"apikey": token}, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                df = pd.DataFrame(data)
                df.rename(columns={'datetime': 'Time', 'temperature': 'Actual_Temp_C', 'wind_speed': 'Actual_WindSpeed'}, inplace=True)
                df['Time'] = pd.to_datetime(df['Time']).dt.tz_localize('UTC')
                df.set_index('Time', inplace=True)
                df = df[['Actual_Temp_C','Actual_WindSpeed']].dropna()
                df['API_Source'] = 'Met Office'
                return df
        except Exception:
            pass
    # Fallback OpenMeteo
    if lat and lon:
        return get_openmeteo_data(lat, lon)
    return pd.DataFrame()

# --- OPSD Enerji Verisi ---
def get_opsd_data() -> pd.DataFrame:
    """
    OPSD dataset: Germany renewable generation (Wind) and load.
    URL örnek: https://data.open-power-system-data.org/time_series/2023-06-30/time_series_60min_singleindex.csv
    """
    url = "https://data.open-power-system-data.org/time_series/2023-06-30/time_series_60min_singleindex.csv"
    try:
        df = pd.read_csv(url, parse_dates=['utc_timestamp'], index_col='utc_timestamp')
        # Örnek: Wind generation MW ve Total Load MW
        df_subset = df[['DE_wind_actual_entsoe_transparency','DE_load_actual_entsoe_transparency']].copy()
        df_subset.rename(columns={
            'DE_wind_actual_entsoe_transparency': 'WindGen_MW',
            'DE_load_actual_entsoe_transparency': 'Target_Actual'
        }, inplace=True)
        return df_subset
    except Exception:
        return pd.DataFrame()

# --- Spot Fiyat Verisi (yFinance) ---
def get_spot_price(ticker="EXC.L") -> pd.DataFrame:
    """
    EXC.L = Elexon / UK Spot Price (örnek)
    """
    try:
        data = yf.download(ticker, period="5d", interval="1h")
        if data.empty: return pd.DataFrame()
        df = pd.DataFrame({'SpotPrice_EUR': data['Close']})
        return df
    except Exception:
        return pd.DataFrame()

# --- run_energy_forecast ---
def run_energy_forecast(city: str) -> pd.DataFrame:
    # 1. Hava durumu
    weather_df = get_metoffice_data(city)
    if weather_df.empty: return pd.DataFrame()
    api_source = weather_df.get('API_Source', 'Bilinmiyor').iloc[0]
    weather_df = weather_df.drop(columns=['API_Source'], errors='ignore')

    # 2. OPSD ve Spot Price
    opsd_df = get_opsd_data()
    spot_df = get_spot_price()
    
    # Merge dataframes
    df_list = [weather_df]
    if not opsd_df.empty: df_list.append(opsd_df)
    if not spot_df.empty: df_list.append(spot_df)
    
    merged_df = pd.concat(df_list, axis=1).sort_index()
    
    # Eksik veriler için forward fill
    merged_df.fillna(method='ffill', inplace=True)
    
    # 3. Basit stacking tahmini (örnek)
    # Final_Stacking_Prediction = Target_Actual * 0.5 + WindGen etkisi + hava etkisi
    if 'Target_Actual' not in merged_df.columns:
        merged_df['Target_Actual'] = np.nan
    if 'WindGen_MW' not in merged_df.columns:
        merged_df['WindGen_MW'] = np.nan
    merged_df['Final_Stacking_Prediction'] = (
        merged_df.get('Target_Actual',0)*0.5 +
        merged_df.get('WindGen_MW',0)*0.01 +
        merged_df['Actual_Temp_C']*2
    )

    # API kaynağını ekle
    merged_df['API_Source'] = api_source
    return merged_df
