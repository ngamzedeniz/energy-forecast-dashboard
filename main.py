import os
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
from datetime import datetime, timedelta

from model_utils import (
    CITIES, get_weather_data, get_elexon_data,
    train_stacking_model, predict_with_stacking, get_meteorological_insight
)

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "cities": CITIES})


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, ticker: str = Form(...), city: str = Form(...)):
    ticker = ticker.upper()
    # --- Weather Data ---
    df_weather, error = get_weather_data(city)
    if error:
        return templates.TemplateResponse("index.html", {"request": request, "cities": CITIES, "error_message": error})

    # --- Meteorological Insight ---
    insight = get_meteorological_insight(df_weather)

    # --- Elexon Data ---
    df_energy, error = get_elexon_data(ticker)
    if error or df_energy is None or df_energy.empty:
        return templates.TemplateResponse("index.html", {"request": request, "cities": CITIES, "error_message": error})

    # --- Features for Model ---
    feature_cols = ["Temperature", "Wind_Speed", "Cloud_Cover", "Precipitation"]
    target_price_col = "Price"  # Elexon dataset örnek
    target_volume_col = "Volume"

    # Train stacking models
    price_model = train_stacking_model(df_energy, target_price_col, feature_cols)
    volume_model = train_stacking_model(df_energy, target_volume_col, feature_cols)

    # Predict next period
    predicted_price = predict_with_stacking(price_model, df_weather, feature_cols)[-1]
    predicted_volume = predict_with_stacking(volume_model, df_weather, feature_cols)[-1]

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "ticker": ticker,
            "city": city,
            "predicted_price": f"{predicted_price:,.2f}",
            "predicted_volume": f"{int(predicted_volume):,}",
            "insight": insight,
            "weather_data": df_weather.to_dict(orient="records")
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
