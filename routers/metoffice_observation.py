import os
import requests
from fastapi import APIRouter, HTTPException
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()
OBSERVATION_API_URL = "https://data.hub.api.metoffice.gov.uk/observation-land/1/nearest"
OBSERVATION_API_KEY = os.getenv("METOFFICE_OBSERVATION_API_KEY")

@router.get("/nearest")
def get_nearest_observation(lat: float, lon: float):
    if not OBSERVATION_API_KEY:
        raise HTTPException(status_code=500, detail="Observation API key missing.")
    headers = {"apikey": OBSERVATION_API_KEY}
    params = {"lat": lat, "lon": lon}
    response = requests.get(OBSERVATION_API_URL, headers=headers, params=params)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    return response.json()
