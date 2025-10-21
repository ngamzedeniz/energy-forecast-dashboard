from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import plotly.express as px
import numpy as np
from model_utils import CITIES, get_land_observation, generate_insight, get_model_predictions

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "cities": CITIES.keys(), "error_message": None})

@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request, ticker: str = Form(...), city: str = Form(...)):
    try:
        obs_df = get_land_observation(city)
        insight = generate_insight(obs_df)

        predicted_price, predicted_volume = get_model_predictions(obs_df)

        temp_plot = px.line(obs_df, x="datetime", y="temperature", title="Temperature Forecast (°C)").to_html(full_html=False)
        anomaly_plot = px.line(obs_df, x="datetime", y="temperature_anomaly", title="Temperature Anomaly (°C)").to_html(full_html=False)
        wind_plot = px.line(obs_df, x="datetime", y="wind_speed", title="Wind Speed (m/s)").to_html(full_html=False)
        price_plot = px.line(x=list(range(10)), y=np.random.randn(10).cumsum()+predicted_price, title="Price Trend").to_html(full_html=False)
        volume_plot = px.line(x=list(range(10)), y=np.random.randn(10).cumsum()+predicted_volume, title="Volume Trend").to_html(full_html=False)

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
        return templates.TemplateResponse("index.html", {"request": request, "cities": CITIES.keys(), "error_message": str(e)})
