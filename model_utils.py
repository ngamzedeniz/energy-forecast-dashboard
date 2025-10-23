# model_utils.py (GÜNCELLENMİŞ VERSİYON)

import pandas as pd
import numpy as np
import os
import requests 
import json
from requests.exceptions import RequestException
import datetime

# --- Şehir Verilerini Yükleme (Lat/Lon almak için gerekli) ---
try:
    with open("cities.json", "r") as f:
        CITIES_DATA = json.load(f)
        CITIES_DICT = {c['name']: c for c in CITIES_DATA}
except FileNotFoundError:
    CITIES_DICT = {}


# --- Yardımcı Fonksiyonlar (Aynı Kalır) ---
def calculate_hdd_cdd(temp_c: float, base_temp: float = 18.0, hdd: bool = True) -> float:
    # ... (Aynı kalmalı)
    if hdd:
        return max(0, base_temp - temp_c)
    return max(0, temp_c - base_temp)

# ... (get_opsd_data ve run_energy_forecast gibi diğer fonksiyonlar aynı kalır)


# --- 1. FONKSİYON: OpenMeteo Yedek API Çağrısı ---

def get_openmeteo_data(lat: float, lon: float) -> pd.DataFrame:
    """OpenMeteo Historical API'den son 5 günlük hava durumu verisini çeker."""
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=5) # Son 5 günü çekiyoruz
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    # temperature_2m ve windspeed_10m çekiyoruz (Met Office'e en yakın değişkenler)
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": "temperature_2m,windspeed_10m",
        "timezone": "auto",
        "temperature_unit": "celsius",
        "wind_speed_unit": "ms" # m/s (Met Office ile aynı)
    }

    print(f"LOG: OpenMeteo API'si deneniyor: Lat={lat}, Lon={lon}")
    
    try:
        resp = requests.get(url, params=params, timeout=15)
        
        if resp.status_code != 200:
            print(f"KRİTİK LOG: OpenMeteo Hata Kodu: {resp.status_code}. Mesaj: {resp.text[:150]}...")
            return pd.DataFrame() 

        data = resp.json()
        
        if 'hourly' not in data:
            print("KRİTİK LOG: OpenMeteo yanıtında 'hourly' verisi bulunamadı.")
            return pd.DataFrame()
            
        hourly_data = data['hourly']
        
        df = pd.DataFrame({
            'Time': pd.to_datetime(hourly_data['time']),
            'Actual_Temp_C': hourly_data['temperature_2m'],
            'Actual_WindSpeed': hourly_data['windspeed_10m']
        })
        
        df.set_index('Time', inplace=True)
        print(f"LOG: OpenMeteo verisi başarıyla çekildi. Satır: {len(df)}")
        return df

    except RequestException as e:
        print(f"KRİTİK LOG: OpenMeteo API isteğinde bağlantı hatası: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"KRİTİK LOG: OpenMeteo verisi işlenirken genel hata: {e}")
        return pd.DataFrame()


# --- 2. FONKSİYON: Birincil API Çağrısı (Met Office) ---

def get_metoffice_data(city: str) -> pd.DataFrame:
    """Met Office'tan veri çekmeyi dener. Başarısız olursa OpenMeteo'yu dener."""
    
    # Şehir verilerini al (geohash, lat, lon)
    city_data = CITIES_DICT.get(city)
    if not city_data:
        print(f"KRİTİK LOG: '{city}' için şehir verileri (geohash/lat/lon) bulunamadı.")
        return pd.DataFrame()
        
    geohash = city_data.get('geohash')
    lat = city_data.get('lat')
    lon = city_data.get('lon')

    # Met Office Denemesi (Öncelikli)
    if geohash and os.getenv('METOFFICE_TOKEN'):
        METOFFICE_TOKEN = os.getenv('METOFFICE_TOKEN')
        url = f"https://data.hub.api.metoffice.gov.uk/observation-land/1/{geohash}"
        params = {"apikey": METOFFICE_TOKEN}
        
        print(f"LOG: Birincil API (Met Office) deneniyor: Geohash={geohash}")

        try:
            resp = requests.get(url, params=params, timeout=10)
            
            if resp.status_code == 200:
                data = resp.json()
                df = pd.DataFrame(data)
                df.rename(columns={'datetime': 'Time', 'temperature': 'Actual_Temp_C', 'wind_speed': 'Actual_WindSpeed'}, inplace=True)
                df['Time'] = pd.to_datetime(df['Time']).dt.tz_localize('UTC')
                df.set_index('Time', inplace=True)
                df = df[['Actual_Temp_C', 'Actual_WindSpeed']].dropna()
                
                print(f"LOG: Met Office verisi başarıyla çekildi. Satır: {len(df)}")
                # Met Office başarılı, veriyi işaretle ve döndür
                df['API_Source'] = 'Met Office'
                return df
                
            else:
                # Met Office hatası varsa logla ve OpenMeteo'ya geçmek için boş döndür
                print(f"KRİTİK LOG: Met Office API Hata Kodu: {resp.status_code}. OpenMeteo'ya geçiliyor.")
                try:
                    print(f"KRİTİK LOG: API Hata Detayı (JSON): {resp.json().get('message', 'Yok')}")
                except:
                    pass
        except RequestException as e:
            print(f"KRİTİK LOG: Met Office API isteğinde bağlantı hatası: {e}. OpenMeteo'ya geçiliyor.")
        except Exception as e:
            print(f"KRİTİK LOG: Met Office verisi işlenirken hata: {e}. OpenMeteo'ya geçiliyor.")

    # --- Yedekleme: OpenMeteo Denemesi ---
    if lat and lon:
        df_openmeteo = get_openmeteo_data(lat, lon)
        if not df_openmeteo.empty:
            df_openmeteo['API_Source'] = 'OpenMeteo'
            return df_openmeteo
            
    print("KRİTİK LOG: Met Office ve OpenMeteo'dan veri alınamadı.")
    return pd.DataFrame() 
    
    
# --- run_energy_forecast (API Kaynağını İşler) ---

def run_energy_forecast(city: str) -> pd.DataFrame:
    """Enerji tahmin modelini çalıştırır ve API kaynağını korur."""
    
    # 1. Hava Durumu Verisi Çekme (Yedekleme burada gerçekleşir)
    weather_df = get_metoffice_data(city)
    
    if weather_df.empty:
        return pd.DataFrame()
        
    # API kaynağını sakla
    api_source = weather_df.get('API_Source', 'Bilinmiyor').iloc[0]
    weather_df = weather_df.drop(columns=['API_Source'], errors='ignore')

    # ... (Kalan tüm model hazırlığı ve çalıştırma adımları aynı kalır)
    
    # (Bu kısım, HDD/CDD hesaplama, OPSD verisi çekme ve model çalıştırma adımlarıdır)
    # Varsayılan değerler ve simülasyonlar kullanıldığını varsayıyoruz.
    
    # Örn:
    # opsd_df = get_opsd_data() 
    # merged_df = pd.merge(opsd_df, weather_df, left_index=True, right_index=True, how='inner')
    
    # Basit bir simülasyon: Son 24 saatin verisi
    sim_data = {
        'Final_Stacking_Prediction': np.random.rand(len(weather_df)) * 100 + 500,
        'Target_Actual': np.random.rand(len(weather_df)) * 100 + 500,
        'SpotPrice_EUR': np.random.rand(len(weather_df)) * 50 + 80,
        'WindGen_MW': np.random.rand(len(weather_df)) * 200 + 1000
    }
    
    # ÖRNEK AMAÇLI BİRLEŞTİRME VE DÖNDÜRME
    df_result = pd.DataFrame(sim_data, index=weather_df.index)
    df_result = df_result.join(weather_df)
    
    # API kaynağını geri ekle
    df_result['API_Source'] = api_source
    return df_result
