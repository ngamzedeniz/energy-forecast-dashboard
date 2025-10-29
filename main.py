from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import pandas as pd
import json
import plotly.express as px
from model_utils import run_energy_forecast
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(title="Energy Forecast Dashboard")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

try:
    with open("cities.json", "r") as f:
        CITIES_DATA = json.load(f)
        CITIES_NAMES = [c['name'] for c in CITIES_DATA]
except FileNotFoundError:
    CITIES_NAMES = ["Hata: cities.json bulunamadı"]
    logging.error("cities.json dosyası bulunamadı!")


def create_plot(df: pd.DataFrame, y_cols: list, title: str, y_label: str) -> str:
    df_plot = df.copy()
    color_map = {
        'Final_Stacking_Prediction': '#0d6efd',
        'Target_Actual': '#dc3545',
        'Actual_Temp_C': '#ffc107',
        'Actual_WindSpeed': '#20c997',
        'SpotPrice_EUR': '#6f42c1',
        'WindGen_MW': '#198754'
    }

    fig = px.line(
        df_plot.reset_index().rename(columns={'index': 'Time'}),
        x='Time',
        y=y_cols,
        title=title,
        labels={'value': y_label, 'Time': 'Time'},
        height=400,
        color_discrete_map={col: color_map.get(col, '#333') for col in y_cols}
    )
    fig.update_layout(legend_title_text='Series', hovermode="x unified")
    return fig.to_html(full_html=False, default_height=400)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "cities": CITIES_NAMES}
    )


@app.post("/forecast", response_class=HTMLResponse)
async def forecast(request: Request, city: str = Form(...)):
    logging.info(f"'{city}' için tahmin isteği alındı.")
    
    results_df = run_energy_forecast(city)
    
    if results_df.empty or 'Actual_Temp_C' not in results_df.columns:
        error_message = (
            "CRITICAL ERROR: Could not retrieve data from primary (Met Office) or backup (OpenMeteo) APIs."
        )
        logging.error("Forecast failed.")
        return templates.TemplateResponse(
            "result.html",
            {"request": request, "city": city, "error": error_message}
        )
    
    api_source = results_df.get('API_Source', pd.Series(['Unknown'])).iloc[0]
    results_df = results_df.drop(columns=['API_Source'], errors='ignore')

    avg_temp = results_df['Actual_Temp_C'].mean()
    avg_wind_speed_actual = results_df['Actual_WindSpeed'].mean()
    max_wind_speed_actual = results_df['Actual_WindSpeed'].max()
    avg_price = results_df['SpotPrice_EUR'].mean()

    interpretation = f"Data Source: {api_source}\n\n"

    if avg_temp < 8:
        interpretation += "• CRITICAL COLD: Heating demand will be very high.\n"
    elif avg_temp < 15:
        interpretation += "• Mild-Cold: Heating will be needed.\n"
    elif avg_temp > 22:
        interpretation += "• HOT WEATHER: Cooling demand will increase.\n"
    else:
        interpretation += "• Normal seasonal conditions.\n"
    
    if avg_wind_speed_actual > 10: 
        interpretation += f"• High Wind: Wind energy generation is strong.\n"
    elif avg_wind_speed_actual < 4:
        interpretation += f"• Low Wind: Backup energy sources needed.\n"
    else:
        interpretation += "• Stable wind conditions.\n"

    demand_plot = create_plot(
        results_df,
        ['Final_Stacking_Prediction', 'Target_Actual'],
        f"{city} Energy Demand Forecast",
        "Demand (Units)"
    )

    weather_plot = create_plot(
        results_df,
        ['Actual_Temp_C', 'Actual_WindSpeed'],
        f"Actual Weather Conditions ({api_source})",
        "Value (°C / m/s)"
    )

    market_plot = create_plot(
        results_df,
        ['WindGen_MW', 'SpotPrice_EUR'],
        "Simulated Energy Market",
        "MW / EUR"
    )

    table_data = results_df.reset_index().rename(columns={'index': 'Time'})
    table_data['Time'] = table_data['Time'].dt.strftime('%Y-%m-%d %H:%M')

    table_rows = table_data[[
        'Time', 'Final_Stacking_Prediction', 'Target_Actual',
        'Actual_Temp_C', 'Actual_WindSpeed', 'SpotPrice_EUR', 'WindGen_MW'
    ]].round(2).to_dict('records')

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "city": city,
            "interpretation": interpretation,
            "avg_temp": f"{avg_temp:.1f}",
            "avg_wind": f"{avg_wind_speed_actual:.1f}",
            "max_wind": f"{max_wind_speed_actual:.1f}",
            "avg_price": f"{avg_price:.2f}",
            "predicted_demand": f"{results_df['Final_Stacking_Prediction'].iloc[-1]:.2f}",
            "demand_plot": demand_plot,
            "weather_plot": weather_plot,
            "market_plot": market_plot,
            "table_rows": table_rows
        }
    )
