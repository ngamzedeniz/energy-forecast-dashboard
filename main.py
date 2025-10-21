from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from model_utils import CITIES, get_land_observation, generate_insight, get_model_predictions
import plotly.express as px

app = FastAPI()
templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "cities": list(CITIES.keys())})


@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request, ticker: str = Form(...), city: str = Form(...)):
    try:
        # Observation verisini al
        obs_df = get_land_observation(city)
        insight = generate_insight(obs_df)

        # ML tahmini
        predicted_price, predicted_volume = get_model_predictions(ticker, obs_df)

        # Görseller
        temp_plot = px.line(obs_df, x="datetime", y="temperature", title="Temperature (°C)").to_html(full_html=False)
        wind_plot = px.line(obs_df, x="datetime", y="wind_speed", title="Wind Speed (m/s)").to_html(full_html=False)

        return templates.TemplateResponse("result.html", {
            "request": request,
            "ticker": ticker,
            "city": city,
            "insight": insight,
            "predicted_price": predicted_price,
            "predicted_volume": predicted_volume,
            "temp_plot": temp_plot,
            "wind_plot": wind_plot
        })

    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "cities": list(CITIES.keys()),
            "error_message": str(e)
        })
