# main.py (güncellenmiş versiyon)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from model_utils import get_weather_data, get_elexon_data, train_stacking_model, predict_with_stacking, get_meteorological_insight

app = FastAPI(title="UK & Scotland Energy Forecast API")

# --- Request model ---
class PredictRequest(BaseModel):
    city: str
    target: Optional[str] = "wind"

# --- Root ---
@app.get("/")
async def root():
    return {"message": "Welcome to the UK & Scotland Energy Forecast API. Use /predict with POST."}

# --- Predict (POST) ---
@app.post("/predict")
async def predict(request: PredictRequest):
    city = request.city
    target = request.target

    # Weather data
    weather_df, err = get_weather_data(city)
    if err:
        raise HTTPException(status_code=400, detail=f"Weather API error: {err}")

    # Example Elexon data (simplified)
    elexon_df, err = get_elexon_data("FUELINST")  # örnek ticker
    if err:
        raise HTTPException(status_code=400, detail=f"Elexon API error: {err}")

    # Feature columns (basit örnek)
    feature_cols = ["Temperature", "Wind_Speed", "Cloud_Cover", "Temp_Anomaly"]
    target_col = "Wind_Speed" if target == "wind" else "Temperature"

    # Train stacking model
    model = train_stacking_model(weather_df, target_col, feature_cols)
    predictions = predict_with_stacking(model, weather_df, feature_cols)
    insight = get_meteorological_insight(weather_df)

    return {
        "city": city,
        "target": target,
        "predictions": predictions.tolist(),
        "insight": insight
    }

# --- Predict (GET) for testing ---
@app.get("/predict")
async def predict_get():
    return {"message": "Use POST method with JSON payload. Example: {'city':'London','target':'wind'}"}
