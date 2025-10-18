import os
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from model_utils import (
    get_weather_data,
    get_elexon_data,
    train_stacking_model,
    predict_with_stacking,
    get_meteorological_insight,
)

app = FastAPI()

# --- Dizinler ---
if not os.path.exists("static"):
    os.makedirs("static")
if not os.path.exists("templates"):
    os.makedirs("templates")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- Ana Sayfa ---
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "cities": list(get_weather_data.__defaults__[0])})

# --- Form Submit / Predict ---
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, city: str = Form(...), ticker: str = Form(...)):
    # Weather Data
    weather_df, weather_error = get_weather_data(city)
    if weather_error:
        return HTMLResponse(f"Weather API error: {weather_error}", status_code=500)

    # Elexon Data
    elexon_df, elexon_error = get_elexon_data(ticker)
    if elexon_error:
        return HTMLResponse(f"Elexon API error: {elexon_error}", status_code=500)

    # Basit örnek feature/target sütunları
    feature_cols = ["Temperature", "Wind_Speed", "Cloud_Cover", "Temp_Anomaly"]
    target_col = "Precipitation"  # örnek
    model = train_stacking_model(weather_df, target_col, feature_cols)
    predictions = predict_with_stacking(model, weather_df, feature_cols)

    insight = get_meteorological_insight(weather_df)

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "city": city,
            "ticker": ticker,
            "predictions": predictions.tolist(),
            "insight": insight
        }
    )

# --- Uvicorn Start Command için ---
# Render start command: uvicorn main:app --host 0.0.0.0 --port $PORT
