# model_utils.py (GERÇEK MET OFFICE VERİSİ İLE GÜNCELLENDİ)

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import json
import os
import requests # Yeni eklenen
from requests.exceptions import RequestException

# Rastgelelik sabiti
np.random.seed(42)

# --- Sabitler ve Veri Yükleme ---
METOFFICE_TOKEN = os.getenv('METOFFICE_TOKEN') # Render ortam değişkeninden okunacak

try:
    with open("cities.json", "r") as f:
        CITIES_DATA_MAP = {c['name']: c for c in json.load(f)}
except FileNotFoundError:
    CITIES_DATA_MAP = {} # Boş bırakılır, hata main.py'de ele alınır.

# --- Yardımcı Fonksiyonlar ---

def calculate_hdd_cdd(temp_c, base_temp=18.0):
    """Isıtma ve Soğutma Derece Günleri hesaplama."""
    if temp_c is None or not isinstance(temp_c, (int, float)):
        return 0.0, 0.0
    hdd = max(0, base_temp - temp_c)
    cdd = max(0, temp_c - base_temp) 
    return hdd, cdd

def get_metoffice_data(geohash: str) -> pd.DataFrame:
    """Met Office Hub Observation-Land API'den gerçek hava durumu verisi çeker."""
    if not METOFFICE_TOKEN:
        print("KRİTİK HATA: METOFFICE_TOKEN ortam değişkeni tanımlı değil!")
        return pd.DataFrame() # Simülasyon yerine boş döndürürüz.
        
    url = f"https://data.hub.api.metoffice.gov.uk/observation-land/1/{geohash}"
    params = {"apikey": METOFFICE_TOKEN}
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status() # HTTP 4xx veya 5xx hatalarını yakalar
        data = resp.json()
        
        # Gelen veriyi DataFrame'e çevir
        df = pd.DataFrame(data)
        
        # Sütunları yeniden adlandır ve temizle
        df.rename(columns={'datetime': 'Time', 'temperature': 'Actual_Temp_C', 'wind_speed': 'Actual_WindSpeed'}, inplace=True)
        df['Time'] = pd.to_datetime(df['Time']).dt.tz_localize('UTC')
        df.set_index('Time', inplace=True)
        
        # Sadece gerekli sütunları alırız (ve varsa NaN'leri atarız)
        df = df[['Actual_Temp_C', 'Actual_WindSpeed']].dropna()
        
        print(f"-> Met Office verisi başarıyla çekildi. Satır: {len(df)}")
        return df
        
    except RequestException as e:
        print(f"API isteğinde hata: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Veri işlenirken hata: {e}")
        return pd.DataFrame()


def get_opsd_data(time_index):
    """OPSD benzeri saatlik simüle edilmiş enerji verisi çeker (Dış veriler için hala simülasyon)."""
    data = pd.DataFrame(index=time_index)
    data['SpotPrice_EUR'] = 30 + 10 * np.sin(np.linspace(0, 2 * np.pi * 5, len(data))) + np.random.randn(len(data)) * 2
    data['WindGen_MW'] = 500 + 100 * np.sin(np.linspace(0, 2 * np.pi * 5, len(data))) + np.random.randn(len(data)) * 50
    return data

# --- Ana Model Fonksiyonu ---

def run_energy_forecast(city_name: str) -> pd.DataFrame:
    """Seçilen şehir için Master DF'i oluşturur, modeli eğitir ve tahmini döndürür."""
    
    city_info = CITIES_DATA_MAP.get(city_name)
    if not city_info or 'geohash' not in city_info:
        return pd.DataFrame() # Şehir bilgisi yoksa boş döndür.

    geohash = city_info['geohash']
    
    # 1. Gerçek Hava Durumu Verisini Çekme
    weather_df = get_metoffice_data(geohash)
    if weather_df.empty:
        return pd.DataFrame() # Hava durumu verisi yoksa dururuz.

    # 2. Master DF'i oluşturma (Hava durumu indeksi kullanılır)
    master_df = weather_df.copy()
    
    # 3. Hesaplanan Özellikler (HDD/CDD, Takvim, Trend)
    master_df['HDD'], master_df['CDD'] = zip(*master_df['Actual_Temp_C'].apply(
        lambda t: calculate_hdd_cdd(t, base_temp=18.0)
    ))
    master_df['dayofweek'] = master_df.index.dayofweek
    master_df['time_step'] = np.arange(len(master_df))

    # 4. OPSD Simülasyon Verisini Çekme ve Birleştirme
    opsd_df = get_opsd_data(master_df.index) 
    master_df = master_df.merge(opsd_df, left_index=True, right_index=True, how='left')
        
    # 5. Temizleme ve Doldurma
    master_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    master_df.ffill(inplace=True) 
    master_df.bfill(inplace=True) # En kötü durumda simülasyon verileri doldurulur
    
    # 6. Target Oluşturma (Bağımlı Değişken, Gerçek Veriye dayalı Simülasyon)
    
    # Şehir Büyüklüğüne göre Talep Çarpanı (Önceki mantık korunur)
    large_cities = ["London", "Manchester", "Birmingham", "Glasgow", "Leeds"]
    medium_cities = ["Edinburgh", "Bristol", "Liverpool", "Newcastle", "Cardiff"]
    
    if city_name in large_cities:
        demand_factor = 1.5 
    elif city_name in medium_cities:
        demand_factor = 1.2
    else:
        demand_factor = 1.0
        
    price_sensitivity = 5
    city_weight = 120 * demand_factor
    
    # Hedef (Target) artık GERÇEK SICAKLIK ve SİMÜLE EDİLMİŞ FİYAT/RÜZGAR verilerine dayanıyor.
    master_df['Target_Demand'] = (master_df['HDD'] * city_weight) + \
                                 (master_df['SpotPrice_EUR'] * price_sensitivity) + \
                                 (master_df['time_step'] * 2) + \
                                 (np.random.randn(len(master_df)) * 50)

    master_df.dropna(subset=['Target_Demand'], how='any', inplace=True) 

    # 7. Modelleme (Aynı Stacking Mantığı)
    target_col = 'Target_Demand'
    if len(master_df) < 50:
        return pd.DataFrame() 

    # Artık 'Actual_Temp_C' model input'u olarak kullanılır
    X = master_df.drop(columns=[target_col]) 
    y = master_df[target_col]
    
    X_train, X_pred, y_train, y_pred = train_test_split(X, y, test_size=0.4, shuffle=False)
    
    # Temel Öğrenenler ve Meta-Model Eğitimi (Aynı kalır)
    models = {
        'Pred_1_XGBoost': XGBRegressor(n_estimators=50, learning_rate=0.08, random_state=42, verbosity=0), 
        'Pred_2_Ridge': Ridge(alpha=1.0), 
        'Pred_3_DecisionTree': DecisionTreeRegressor(max_depth=5, random_state=42) 
    }
    
    predictions_df = X_pred.copy()
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions_df[name] = model.predict(X_pred)

    meta_features = list(models.keys())
    meta_model = LinearRegression()
    
    meta_model.fit(predictions_df[meta_features], y_pred)
    predictions_df['Final_Stacking_Prediction'] = meta_model.predict(predictions_df[meta_features])
    predictions_df['Target_Actual'] = y_pred
    
    # Dönüş: Model tahminleri ve gerçek hava durumu verileri
    return predictions_df[['Final_Stacking_Prediction', 'Target_Actual', 'Actual_Temp_C', 'SpotPrice_EUR', 'WindGen_MW', 'Actual_WindSpeed']].copy()
