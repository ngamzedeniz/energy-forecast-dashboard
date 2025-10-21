import os
import requests
from fastapi import APIRouter, HTTPException
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()
MODEL_API_URL = "https://data.hub.api.metoffice.gov.uk/forecast/model/1/point"
MODEL_API_KEY = os.getenv("METOFFICE_MODEL_API_KEY")

@router.get("/point")
def get_model_forecast(lat: float, lon: float):
    if not MODEL_API_KEY:
        raise HTTPException(status_code=500, detail="Model API key missing.")
    headers = {"apikey": MODEL_API_KEY}
    params = {"lat": lat, "lon": lon}
    response = requests.get(MODEL_API_URL, headers=headers, params=params)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    return response.json()
