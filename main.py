from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional
from model_utils import (
    get_weather_data,
    get_elexon_data,
    train_stacking_model,
    predict_with_stacking,
    get_meteorological_insight
)

app = FastAPI(title="UK & Scotland Energy Forecast API")
templates = Jinja2Templates(directory="templates")

# --- HTML sayfa ---
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    cities = ["London", "Edinburgh", "Glasgow", "Aberdeen", "Inverness"]
    return templates.TemplateResponse("index.html", {"request": request, "cities": cities})


# --- HTML formdan tahmin ---
@app.post("/predict", response_class=HTMLResponse)
async def predict_form(
    request: Request,
    city: str = Form(...),
    ticker: Optional[str] = Form(None)
):
    try:
        weather_df, err = get_weather_data(city)
        if err:
            raise HTTPException(status_code=400, detail=f"Weather API error: {err}")

        elexon_df, err = get_elexon_data("FUELINST")
        if err:
            raise HTTPException(status_code=400, detail=f"Elexon API error: {err}")

        feature_cols = ["Temperature", "Wind_Speed", "Cloud_Cover", "Temp_Anomaly"]
        target_col = "Wind_Speed"

        model = train_stacking_model(weather_df, target_col, feature_cols)
        predictions = predict_with_stacking(model, weather_df, feature_cols)
        insight = get_meteorological_insight(weather_df)

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "cities": ["London", "Edinburgh", "Glasgow", "Aberdeen", "Inverness"],
                "error_message": f"✅ Forecast completed for {ticker or 'default'} in {city}.",
                "insight": insight
            }
        )

    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "cities": ["London", "Edinburgh", "Glasgow", "Aberdeen", "Inverness"],
                "error_message": f"⚠️ Error: {str(e)}"
            }
        )


# --- JSON API (örnek: POST /api/predict) ---
class PredictRequest(BaseModel):
    city: str
    target: Optional[str] = "wind"

@app.post("/api/predict")
async def predict_api(request: PredictRequest):
    city = request.city
    target = request.target

    weather_df, err = get_weather_data(city)
    if err:
        raise HTTPException(status_code=400, detail=f"Weather API error: {err}")

    elexon_df, err = get_elexon_data("FUELINST")
    if err:
        raise HTTPException(status_code=400, detail=f"Elexon API error: {err}")

    feature_cols = ["Temperature", "Wind_Speed", "Cloud_Cover", "Temp_Anomaly"]
    target_col = "Wind_Speed" if target == "wind" else "Temperature"

    model = train_stacking_model(weather_df, target_col, feature_cols)
    predictions = predict_with_stacking(model, weather_df, feature_cols)
    insight = get_meteorological_insight(weather_df)

    return {
        "city": city,
        "target": target,
        "predictions": predictions.tolist(),
        "insight": insight
    }
