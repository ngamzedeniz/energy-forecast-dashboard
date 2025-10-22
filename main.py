# main.py (GÜNCELLENMİŞ: GERÇEK MET OFFICE VERİSİNİ KULLANIR)

from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import pandas as pd
import json
import plotly.express as px
from model_utils import run_energy_forecast # Yeni model_utils'u içe aktar

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
    CITIES_NAMES = ["Hata: cities.json bulunamadı"]

# --- Yardımcı Fonksiyon: Plotly Grafiği Oluşturma ---

def create_plot(df: pd.DataFrame, y_cols: list, title: str, y_label: str) -> str:
    """Plotly ile interaktif çizgi grafik oluşturur ve HTML olarak döndürür."""
    # ... (Bu fonksiyon önceki main.py'dekiyle aynı kalır)
    df_plot = df.copy()
    color_map = {'Final_Stacking_Prediction': '#0d6efd', 'Target_Actual': '#dc3545', 'Actual_Temp_C': '#ffc107', 'Actual_WindSpeed': '#198754', 'SpotPrice_EUR': '#6f42c1', 'WindGen_MW': '#4CAF50'}
    
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
    developer_name = "Oğuzhan Kıyak"
    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "cities": CITIES_NAMES, "developer": developer_name}
    )

@app.post("/forecast", response_class=HTMLResponse)
async def forecast(request: Request, city: str = Form(...)):
    developer_name = "Oğuzhan Kıyak"
    
    # 1. Model Tahminini Çalıştırma
    results_df = run_energy_forecast(city)
    
    if results_df.empty or 'Actual_Temp_C' not in results_df.columns:
        # Hata durumunu veya Met Office tokenının eksik olduğunu belirt
        error_message = "Met Office API'sinden veri alınamadı. Lütfen METOFFICE_TOKEN ortam değişkenini kontrol edin veya API kotanızı/izinlerinizi doğrulayın."
        return templates.TemplateResponse(
            "result.html", 
            {"request": request, "city": city, "error": error_message, "developer": developer_name}
        )

    # 2. Meteorolojik Insight ve Yorumlama (Artık Gerçek Verilere Dayalı)
    avg_temp = results_df['Actual_Temp_C'].mean()
    avg_wind_speed_actual = results_df['Actual_WindSpeed'].mean() # Gerçek rüzgar hızı
    max_wind_speed_actual = results_df['Actual_WindSpeed'].max()
    avg_price = results_df['SpotPrice_EUR'].mean() # Simüle fiyat
    
    # Gelişmiş Yorumlama
    interpretation = "Enerji Talep ve Hava Durumu Değerlendirmesi:\n\n"
    
    if avg_temp < 8:
        interpretation += "• **KRİTİK SOĞUK:** Ortalama sıcaklıklar çok düşük, ısıtma talebi çok yüksek olacaktır.\n"
    elif avg_temp < 15:
        interpretation += "• **Soğuk/Ilıman:** Sıcaklıklar ısıtma ihtiyacını doğuruyor. Kış talebi etken.\n"
    elif avg_temp > 22:
        interpretation += "• **YAZ TALEBİ:** Yüksek sıcaklıklar klima/soğutma talebini ciddi oranda artıracaktır.\n"
    else:
        interpretation += "• **Normal Koşullar:** Sıcaklıklar mevsim normallerinde. Talep, ağırlıklı ticari ve fiyat etkisine bağlı.\n"
        
    if avg_wind_speed_actual > 10: # Örneğin 10 m/s üstü yüksek rüzgar sayılabilir
        interpretation += f"• **Yüksek Rüzgar Hızı:** Gerçek rüzgar hızı ({avg_wind_speed_actual:.1f} m/s) yüksek. Bu, simüle edilmiş rüzgar enerjisi üretimini (WindGen_MW) destekleyerek fiyat baskısını azaltacaktır.\n"
    elif avg_wind_speed_actual < 4:
        interpretation += f"• **Düşük Rüzgar Hızı:** Rüzgar hızı düşük. Şebeke dengelemesi ve yedek enerji kaynaklarına ihtiyaç duyulacaktır.\n"
    else:
        interpretation += "• **Orta Rüzgar:** Rüzgar koşulları stabil.\n"

    # 3. Grafik Oluşturma
    
    demand_plot = create_plot(
        results_df, 
        y_cols=['Final_Stacking_Prediction', 'Target_Actual'], 
        title=f"{city} Enerji Talebi Tahmini", 
        y_label="Talep (Birim)"
    )
    
    # Gerçek Hava Durumu Grafiği (Sıcaklık ve Rüzgar Hızı)
    weather_plot = create_plot(
        results_df, 
        y_cols=['Actual_Temp_C', 'Actual_WindSpeed'], 
        title="Gerçek Met Office Hava Durumu (Sıcaklık ve Rüzgar Hızı)", 
        y_label="Değer (°C / m/s)"
    )
    
    # Enerji Piyasası Grafiği (Simülasyon)
    market_plot = create_plot(
        results_df, 
        y_cols=['WindGen_MW', 'SpotPrice_EUR'], 
        title="Simüle Edilmiş Enerji Piyasası (Üretim ve Fiyat)", 
        y_label="Değer (MW / EUR)"
    )

    # 4. Detaylı Tablo Verisi
    table_data = results_df.reset_index().rename(columns={'index': 'Time'})
    table_data['Time'] = table_data['Time'].dt.strftime('%Y-%m-%d %H:%M')
    table_rows = table_data[[
        'Time', 'Final_Stacking_Prediction', 'Target_Actual', 
        'Actual_Temp_C', 'Actual_WindSpeed', 'SpotPrice_EUR', 'WindGen_MW'
    ]].round(2).to_dict('records')


    # 5. Sonuçları Şablona Gönderme
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
            "weather_plot": weather_plot, # Yeni grafik
            "market_plot": market_plot,   # Yeni grafik
            "table_rows": table_rows,
            "developer": developer_name
        }
    )
