from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import pandas as pd
from model_utils import forecast_energy, get_weather_insight

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# UK + Scotland critical cities (wind & solar)
CITY_COORDINATES = {
    "London": {"lat":51.5074,"lon":-0.1278},
    "Manchester":{"lat":53.4808,"lon":-2.2426},
    "Birmingham":{"lat":52.4862,"lon":-1.8904},
    "Leeds":{"lat":53.8008,"lon":-1.5491},
    "Glasgow":{"lat":55.8642,"lon":-4.2518},
    "Edinburgh":{"lat":55.9533,"lon":-3.1883},
    "Aberdeen":{"lat":57.1497,"lon":-2.0943},
    "Dundee":{"lat":56.4620,"lon":-2.9707},
    "Inverness":{"lat":57.4778,"lon":-4.2247},
    "Belfast":{"lat":54.5973,"lon":-5.9301}
}
CITIES = list(CITY_COORDINATES.keys())


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "cities": CITIES})


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, ticker: str = Form(...), city: str = Form(...)):
    if city not in CITY_COORDINATES:
        return templates.TemplateResponse("index.html", {"request": request, "cities": CITIES, "error_message":"Invalid city selected."})
    
    df_weather, avg_wind, max_wind, interpretation = get_weather_insight(CITY_COORDINATES[city])
    predicted_price, predicted_volume = forecast_energy(ticker, df_weather)

    return templates.TemplateResponse("result.html", {
        "request": request,
        "ticker": ticker,
        "city": city,
        "predicted_price": f"{predicted_price:.2f}" if predicted_price else "N/A",
        "predicted_volume": f"{predicted_volume:.0f}" if predicted_volume else "N/A",
        "avg_wind": avg_wind,
        "max_wind": max_wind,
        "interpretation": interpretation,
        "weather_table": df_weather.to_dict('records') if df_weather is not None else []
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
