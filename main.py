# main.py

from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import pandas as pd
import json
import plotly.express as px
from model_utils import run_energy_forecast

app = FastAPI(title="Energy Forecast Dashboard")

# Statik dosyalar ve Jinja2 şablon motoru ayarları
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Şehir verilerini yükleme
try:
    with open("cities.json", "r") as f:
        CITIES_DATA = json.load(f)
        CITIES_NAMES = [c['name'] for c in CITIES_DATA]
except FileNotFoundError:
    CITIES_NAMES = ["London", "Edinburgh", "Berlin"] # Yedek liste

# --- Yardımcı Fonksiyon: Plotly Grafiği Oluşturma ---

def create_plot(df: pd.DataFrame, y_cols: list, title: str, y_label: str) -> str:
    """Plotly ile interaktif çizgi grafik oluşturur ve HTML olarak döndürür."""
    df_plot = df.copy()
    
    # Renk sırasını ayarla (Tahmini öne çıkar)
    color_map = {'Final_Stacking_Prediction': '#0d6efd', 'Target_Actual': '#dc3545', 'Simulated_Temp_C': '#ffc107', 'WindGen_MW': '#198754', 'SpotPrice_EUR': '#6f42c1'}
    
    fig = px.line(
        df_plot.reset_index().rename(columns={'index': 'Time'}), 
        x='Time', 
        y=y_cols, 
        title=title,
        labels={'value': y_label, 'Time': 'Zaman'},
        height=400,
        color_discrete_map={col: color_map.get(col, '#333') for col in y_cols}
    )
    fig.update_layout(legend_title_text='Seri', hovermode="x unified")
    
    return fig.to_html(full_html=False, default_height=400)


# --- API Uç Noktaları ---

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Ana sayfa: Şehir seçim formunu gösterir."""
    developer_name = "Oğuzhan Kıyak"
    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "cities": CITIES_NAMES, "developer": developer_name}
    )

@app.post("/forecast", response_class=HTMLResponse)
async def forecast(request: Request, city: str = Form(...)):
    """Tahmini çalıştırır ve sonuçları dashboard'da gösterir."""
    developer_name = "Oğuzhan Kıyak"
    
    # 1. Model Tahminini Çalıştırma
    results_df = run_energy_forecast(city)
    
    if results_df.empty:
        return templates.TemplateResponse(
            "result.html", 
            {"request": request, "city": city, "error": "Model için yeterli veri oluşturulamadı veya temizlik sonrası veri kalmadı.", "developer": developer_name}
        )

    # 2. Meteorolojik Insight ve Yorumlama
    avg_temp = results_df['Simulated_Temp_C'].mean()
    avg_wind = results_df['WindGen_MW'].mean()
    max_wind = results_df['WindGen_MW'].max()
    avg_price = results_df['SpotPrice_EUR'].mean()
    
    # Gelişmiş Yorumlama
    interpretation = "Enerji Talep ve Hava Durumu Değerlendirmesi:\n\n"
    
    if avg_temp < 10:
        interpretation += "• **Soğuk Hava Uyarısı:** Ortalama sıcaklıkların düşük olması ($\le 10^\circ C$), Isıtma Derece Günlerini (HDD) önemli ölçüde artırarak elektrik talebini yukarı çekecektir.\n"
    elif avg_temp > 20:
        interpretation += "• **Sıcak Hava Uyarısı:** Ortalama sıcaklıkların yüksek olması ($\ge 20^\circ C$), Soğutma Derece Günlerini (CDD) artırarak klima kullanımına bağlı talebi yükseltecektir.\n"
    else:
        interpretation += "• **Ilıman Koşullar:** Sıcaklıklar mevsim normallerinde seyrediyor. Talep, ağırlıklı olarak ticari/endüstriyel aktivite ve enerji fiyatlarına bağlı olacaktır.\n"
        
    if avg_wind > 650:
        interpretation += f"• **Yüksek Rüzgar Üretimi:** Ortalama rüzgar üretimi ($> 650MW$) yüksek. Bu, şebekeye bol miktarda yenilenebilir enerji sağlayarak spot fiyatlar üzerinde **aşağı yönlü** bir baskı oluşturabilir.\n"
    elif avg_wind < 400:
        interpretation += f"• **Düşük Rüzgar Üretimi:** Rüzgar üretimi düşük ($< 400MW$). Yenilenebilir enerjideki bu düşüş, fiyatları ve fosil yakıtlara olan talebi **artırabilir**.\n"
    else:
        interpretation += "• **Normal Rüzgar:** Rüzgar enerjisi üretimi normal seviyelerde. Piyasa dengeli.\n"

    # 3. Grafik Oluşturma
    
    demand_plot = create_plot(
        results_df, 
        y_cols=['Final_Stacking_Prediction', 'Target_Actual'], 
        title=f"{city} Enerji Talebi Tahmini", 
        y_label="Talep (Birim)"
    )
    
    temp_plot = create_plot(
        results_df, 
        y_cols=['Simulated_Temp_C'], 
        title="Simüle Edilmiş Sıcaklık Değişimi", 
        y_label="Sıcaklık (°C)"
    )
    
    wind_price_plot = create_plot(
        results_df, 
        y_cols=['WindGen_MW', 'SpotPrice_EUR'], 
        title="Rüzgar Üretimi ve Spot Fiyat İlişkisi", 
        y_label="Değer (MW / EUR)"
    )

    # 4. Detaylı Tablo Verisi
    table_data = results_df.reset_index().rename(columns={'index': 'Time'})
    table_data['Time'] = table_data['Time'].dt.strftime('%Y-%m-%d %H:%M')
    table_rows = table_data[[
        'Time', 'Final_Stacking_Prediction', 'Target_Actual', 
        'Simulated_Temp_C', 'SpotPrice_EUR', 'WindGen_MW'
    ]].round(2).to_dict('records')


    # 5. Sonuçları Şablona Gönderme
    return templates.TemplateResponse(
        "result.html", 
        {
            "request": request,
            "city": city,
            "interpretation": interpretation,
            "avg_temp": f"{avg_temp:.1f}",
            "avg_wind": f"{avg_wind:.1f}",
            "max_wind": f"{max_wind:.1f}",
            "avg_price": f"{avg_price:.2f}",
            "predicted_demand": f"{results_df['Final_Stacking_Prediction'].iloc[-1]:.2f}",
            "demand_plot": demand_plot,
            "temp_plot": temp_plot,
            "wind_price_plot": wind_price_plot,
            "table_rows": table_rows,
            "developer": developer_name
        }
    )
