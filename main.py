from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional
from model_utils import CITIES, get_weather_data, get_model_predictions, get_meteorological_insight

app = FastAPI(title="UK & Scotland Energy Forecast API")
templates = Jinja2Templates(directory="templates")

# --- HTML sayfa ---
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "cities": CITIES})

# --- HTML formdan tahmin ---
@app.post("/predict", response_class=HTMLResponse)
async def predict_form(
    request: Request,
    city: str = Form(...),
    target: str = Form("Wind_Speed")
):
    df_weather, err = await get_weather_data(city)
    if err:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "cities": CITIES, "error_message": f"⚠️ Weather API error: {err}"}
        )

    preds = await get_model_predictions(df_weather, target_col=target)
    insight = get_meteorological_insight(df_weather)

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "city": city,
            "target": target,
            "weather_data": df_weather.to_dict(orient="records"),
            "predictions": preds.tolist(),
            "insight": insight["interpretation"]
        }
    )

# --- JSON API ---
class PredictRequest(BaseModel):
    city: str
    target: Optional[str] = "Wind_Speed"

@app.post("/api/predict")
async def predict_api(request: PredictRequest):
    df_weather, err = await get_weather_data(request.city)
    if err:
        raise HTTPException(status_code=400, detail=f"Weather API error: {err}")

    preds = await get_model_predictions(df_weather, target_col=request.target)
    insight = get_meteorological_insight(df_weather)
    return {
        "city": request.city,
        "target": request.target,
        "predictions": preds.tolist(),
        "insight": insight
    }
