from fastapi import FastAPI, Query
from pydantic import BaseModel
import pandas as pd
import requests
import os
from datetime import datetime, timedelta
from model_utils import train_stacking_model, forecast_generation
from dotenv import load_dotenv

# --- ENV AYARLARI ---
load_dotenv()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
ELEXON_API_KEY = os.getenv("ELEXON_API_KEY", "0fad4fex2qqke42")

app = FastAPI(title="UK Wind Energy Forecast API",
              description="Scottish wind energy & generation forecasting using stacking models.",
              version="1.0.0")

# --- İSKOÇYA ŞEHİRLERİ ---
SCOTLAND_CITIES = {
    "Glasgow": {"lat": 55.8642, "lon": -4.2518},
    "Edinburgh": {"lat": 55.9533, "lon": -3.1883},
    "Aberdeen": {"lat": 57.1497, "lon": -2.0943},
    "Inverness": {"lat": 57.4778, "lon": -4.2247},
    "Dundee": {"lat": 56.4620, "lon": -2.9707},
    "Perth": {"lat": 56.3952, "lon": -3.4314},
    "Ayr": {"lat": 55.4586, "lon": -4.6292},
    "Dumfries": {"lat": 55.0690, "lon": -3.6110},
    "Stirling": {"lat": 56.1165, "lon": -3.9369}
}


# --- ELEXON API'DEN ELEKTRİK ÜRETİM VERİSİ ÇEKME ---
def fetch_elexon_data():
    """Son 24 saatlik UK elektrik üretimi verisi (MW)"""
    try:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=24)
        url = (
            f"https://api.bmreports.com/BMRS/B1620/v1?"
            f"APIKey={ELEXON_API_KEY}&SettlementDate={start_time.date()}&Period=1&ServiceType=csv"
        )
        r = requests.get(url)
        if r.status_code == 200 and 'csv' in r.text:
            df = pd.read_csv(pd.compat.StringIO(r.text))
            df = df.rename(columns=lambda x: x.strip())
            df = df[['Settlement Date', 'Quantity']]
            df.columns = ['timestamp', 'generation_mw']
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.dropna()
            return df
        else:
            return pd.DataFrame()
    except Exception as e:
        print("Elexon fetch error:", e)
        return pd.DataFrame()


# --- OPENWEATHER API'DEN HAVA VERİSİ ÇEKME ---
def fetch_weather_data(city_name):
    coords = SCOTLAND_CITIES.get(city_name)
    if not coords:
        raise ValueError("Invalid city name")

    lat, lon = coords['lat'], coords['lon']
    url = (
        f"https://api.openweathermap.org/data/2.5/forecast?"
        f"lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    )
    r = requests.get(url)
    data = r.json()

    df = pd.DataFrame([
        {
            "timestamp": pd.to_datetime(entry["dt_txt"]),
            "wind_speed": entry["wind"]["speed"],
            "temperature": entry["main"]["temp"],
            "pressure": entry["main"]["pressure"],
            "clouds": entry["clouds"]["all"]
        }
        for entry in data["list"][:16]  # 48 saatlik tahmin
    ])
    return df


# --- MODEL NESNESİ (GLOBAL) ---
trained_model = None
model_score = None


# --- /train ENDPOINT ---
@app.get("/train")
def train_model(city: str = Query("Glasgow", description="Scottish city name")):
    global trained_model, model_score

    weather_df = fetch_weather_data(city)
    elexon_df = fetch_elexon_data()

    if elexon_df.empty:
        return {"error": "Elexon data unavailable"}

    # basit zaman eşleştirmesi
    merged = pd.merge_asof(weather_df.sort_values("timestamp"),
                           elexon_df.sort_values("timestamp"),
                           on="timestamp", direction="nearest")

    merged.dropna(inplace=True)
    trained_model, model_score = train_stacking_model(merged)

    return {
        "message": f"Model trained successfully for {city}.",
        "score": round(model_score, 3),
        "data_points": len(merged)
    }


# --- /predict ENDPOINT ---
class WeatherInput(BaseModel):
    wind_speed: float
    temperature: float
    pressure: float
    clouds: float


@app.post("/predict")
def predict_energy(input_data: WeatherInput):
    if trained_model is None:
        return {"error": "Model not trained yet. Call /train first."}

    prediction = forecast_generation(trained_model, input_data.dict())
    return {
        "predicted_generation_mw": prediction,
        "input": input_data.dict()
    }


# --- /status ENDPOINT ---
@app.get("/status")
def status():
    return {
        "model_trained": trained_model is not None,
        "model_score": round(model_score, 3) if model_score else None
    }


# --- LOCAL RUN ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
