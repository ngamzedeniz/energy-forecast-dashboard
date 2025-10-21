from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from model_utils import CITIES, get_land_observation, generate_insight
import plotly.express as px
import pandas as pd
import numpy as np

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "cities": CITIES, "error_message": None})

@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request, ticker: str = Form(...), city: str = Form(...)):
    try:
        obs_df = get_land_observation(city)
        insight = generate_insight(obs_df)

        # Dummy predicted price/volume
        predicted_price = round(100 + np.random.randn()*5, 2)
        predicted_volume = int(1000 + np.random.randn()*50)

        # Plotly charts
        temp_plot = px.line(obs_df, x="datetime", y="temperature", title="Temperature Forecast (°C)").to_html(full_html=False)
        anomaly_plot = px.line(obs_df, x="datetime", y="temperature_anomaly", title="Temperature Anomaly (°C)").to_html(full_html=False)
        wind_plot = px.line(obs_df, x="datetime", y="wind_speed", title="Wind Speed (m/s)").to_html(full_html=False)
        price_plot = px.line(x=list(range(10)), y=100+np.random.randn(10).cumsum(), title="Price Trend").to_html(full_html=False)
        volume_plot = px.line(x=list(range(10)), y=1000+np.random.randn(10).cumsum(), title="Volume Trend").to_html(full_html=False)

        return templates.TemplateResponse("result.html", {
            "request": request,
            "ticker": ticker,
            "city": city,
            "predicted_price": predicted_price,
            "predicted_volume": predicted_volume,
            "avg_wind_speed": insight["avg_wind"],
            "max_wind_text": insight["max_wind"],
            "norm_temp": insight["norm_temp"],
            "detailed_table_data": insight["table"],
            "temp_plot": temp_plot,
            "anomaly_plot": anomaly_plot,
            "wind_plot": wind_plot,
            "price_plot": price_plot,
            "volume_plot": volume_plot
        })
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "cities": CITIES, "error_message": str(e)})
