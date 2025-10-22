# main.py
import os
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
import requests
from model_utils import train_models, predict_models

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

METOFFICE_API_KEY = os.getenv("METOFFICE_API_KEY")
ELEXON_API_KEY = os.getenv("ELEXON_API_KEY")

# Örnek sabit şehirler
CITY_COORDINATES = {
    "London": {"lat": 51.5074, "lon": -0.1278},
    "Birmingham": {"lat": 52.4862, "lon": -1.8904},
    "Glasgow": {"lat": 55.8642, "lon": -4.2518},
}

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "cities": list(CITY_COORDINATES.keys())})

@app.post("/forecast", response_class=HTMLResponse)
async def forecast(request: Request, city: str = Form(...)):
    coords = CITY_COORDINATES[city]

    # MetOffice Observation API
    met_url = f"https://api.metoffice.gov.uk/observation-land/1/nearest?lat={coords['lat']}&lon={coords['lon']}"
    met_headers = {"x-api-key": METOFFICE_API_KEY}
    met_response = requests.get(met_url, headers=met_headers).json()

    # Elexon API (örnek)
    elexon_url = "https://api.bmreports.com/BMRS/FUELINSTHOURS/v1"
    elexon_headers = {"x-api-key": ELEXON_API_KEY}
    elexon_response = requests.get(elexon_url, headers=elexon_headers).json()

    # Dummy veri (model için örnek)
    df = pd.DataFrame({
        "wind_speed": np.random.rand(10),
        "temperature": np.random.rand(10)*30,
        "price": np.random.rand(10)*100
    })

    trained_models, scaler, X_test, y_test = train_models(df, "price")
    predictions = predict_models(trained_models, scaler, df.drop(columns=["price"]))

    return templates.TemplateResponse("result.html", {
        "request": request,
        "ticker": "UK Energy",
        "city": city,
        "predicted_price": round(predictions["RandomForest"][-1], 2),
        "predicted_volume": 100,  # placeholder
        "avg_wind_speed": round(df["wind_speed"].mean(), 2),
        "max_wind_text": round(df["wind_speed"].max(), 2),
        "norm_temp": round(df["temperature"].mean(), 1),
        "interpretation": "Sample interpretation",
        "temp_plot": "<div>Temp Plot Placeholder</div>",
        "anomaly_plot": "<div>Anomaly Plot Placeholder</div>",
        "wind_plot": "<div>Wind Plot Placeholder</div>",
        "detailed_table_data": list(df.itertuples(index=False)),
        "price_plot": "<div>Price Plot Placeholder</div>",
        "volume_plot": "<div>Volume Plot Placeholder</div>"
    })
