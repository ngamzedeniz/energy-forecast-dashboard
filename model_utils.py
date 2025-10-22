# model_utils.py

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import json

# Rastgelelik sabiti
np.random.seed(42)

# Şehir verilerini yükleme
try:
    with open("cities.json", "r") as f:
        CITIES_DATA_MAP = {c['name']: c for c in json.load(f)}
except FileNotFoundError:
    CITIES_DATA_MAP = {"Default": {"name": "Default", "lat": 51.5, "lon": -0.1, "zone": "UK"}}


# --- Yardımcı Fonksiyonlar ---

def calculate_hdd_cdd(temp_c, base_temp=18.0):
    """Isıtma ve Soğutma Derece Günleri hesaplama."""
    if temp_c is None or not isinstance(temp_c, (int, float)):
        return 0.0, 0.0
    hdd = max(0, base_temp - temp_c)
    cdd = max(0, temp_c - base_temp) 
    return hdd, cdd

def get_opsd_data(time_index):
    """OPSD benzeri saatlik simüle edilmiş enerji verisi çeker."""
    data = pd.DataFrame(index=time_index)
    # Fiyat ve Rüzgar Üretimi simülasyonu
    data['SpotPrice_EUR'] = 30 + 10 * np.sin(np.linspace(0, 2 * np.pi * 5, len(data))) + np.random.randn(len(data)) * 2
    data['WindGen_MW'] = 500 + 100 * np.sin(np.linspace(0, 2 * np.pi * 5, len(data))) + np.random.randn(len(data)) * 50
    return data

# --- Ana Model Fonksiyonu ---

def run_energy_forecast(city_name: str) -> pd.DataFrame:
    """Seçilen şehir için Master DF'i oluşturur, modeli eğitir ve tahmini döndürür."""
    
    city_info = CITIES_DATA_MAP.get(city_name, CITIES_DATA_MAP['Default'])
    latitude = city_info['lat']
    
    # 1. Master Time Index Oluşturma
    time_index = pd.date_range(end=datetime.now(), periods=5*24, freq='h').tz_localize('UTC')
    master_df = pd.DataFrame(index=time_index)
    
    # 2. Meteorolojik Özellikler (ŞEHRE DUYARLI SİMÜLASYON)
    
    # Enlem sıcaklık ofseti: UK için 50-60 lat aralığında sıcaklık bazını belirle
    temp_base = 25 - 0.25 * latitude 
    temp_offset = np.sin(np.linspace(0, 2 * np.pi * 5, len(master_df))) * 5 + temp_base
    
    master_df['Simulated_Temp_C'] = temp_offset + np.random.randn(len(master_df)) * 0.5
    
    master_df['HDD'], master_df['CDD'] = zip(*master_df['Simulated_Temp_C'].apply(
        lambda t: calculate_hdd_cdd(t, base_temp=18.0)
    ))
    master_df['dayofweek'] = master_df.index.dayofweek
    master_df['time_step'] = np.arange(len(master_df))

    # 3. OPSD Simülasyon Verisini Çekme ve Birleştirme
    opsd_df = get_opsd_data(time_index) 
    master_df = master_df.merge(opsd_df, left_index=True, right_index=True, how='left')
        
    # 4. Temizleme ve Doldurma
    master_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    master_df.ffill(inplace=True) 
    master_df.bfill(inplace=True)
    
    # 5. Target Oluşturma (ŞEHRE DUYARLI TALEP BAZI)
    
    # Şehir Nüfusu/Büyüklüğüne göre Talep Çarpanı
    large_cities = ["London", "Manchester", "Birmingham", "Glasgow", "Leeds"]
    medium_cities = ["Edinburgh", "Bristol", "Liverpool", "Newcastle", "Cardiff"]
    
    if city_name in large_cities:
        demand_factor = 1.5 
    elif city_name in medium_cities:
        demand_factor = 1.2
    else:
        demand_factor = 1.0 # Diğer küçük şehirler
        
    price_sensitivity = 5
    city_weight = 120 * demand_factor
    
    master_df['Target_Demand'] = (master_df['HDD'] * city_weight) + \
                                 (master_df['SpotPrice_EUR'] * price_sensitivity) + \
                                 (master_df['time_step'] * 2) + \
                                 (np.random.randn(len(master_df)) * 50)

    master_df.dropna(subset=['Target_Demand'], how='any', inplace=True) 

    # 6. Modelleme
    target_col = 'Target_Demand'
    if len(master_df) < 50:
        return pd.DataFrame() 

    X = master_df.drop(columns=[target_col, 'Simulated_Temp_C']) 
    y = master_df[target_col]
    
    X_train, X_pred, y_train, y_pred = train_test_split(X, y, test_size=0.4, shuffle=False)
    
    # Temel Öğrenenler
    models = {
        'Pred_1_XGBoost': XGBRegressor(n_estimators=50, learning_rate=0.08, random_state=42, verbosity=0), 
        'Pred_2_Ridge': Ridge(alpha=1.0), 
        'Pred_3_DecisionTree': DecisionTreeRegressor(max_depth=5, random_state=42) 
    }
    
    predictions_df = X_pred.copy()
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions_df[name] = model.predict(X_pred)

    # Meta-Model (Stacking)
    meta_features = list(models.keys())
    meta_model = LinearRegression()
    
    meta_model.fit(predictions_df[meta_features], y_pred)
    predictions_df['Final_Stacking_Prediction'] = meta_model.predict(predictions_df[meta_features])
    
    predictions_df['Target_Actual'] = y_pred
    
    # Nihai DF'i döndür
    return predictions_df[['Final_Stacking_Prediction', 'Target_Actual', 'Simulated_Temp_C', 'SpotPrice_EUR', 'WindGen_MW']].copy()
