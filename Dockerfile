# 1. Python 3.9 tabanlı resmi imajı kullan
FROM python:3.9

# 2. Çalışma dizinini oluştur
WORKDIR /code

# 3. Önce sadece requirements dosyasını kopyala (Önbellek performansı için)
COPY ./requirements.txt /code/requirements.txt

# 4. Kütüphaneleri yükle (Cache klasörü oluşturmadan)
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 5. Kalan tüm proje dosyalarını (main.py, templates, static vb.) kopyala
COPY . /code

# 6. İzinleri ayarla (Hugging Face'in hata vermemesi için garanti yöntem)
RUN chmod -R 777 /code

# 7. Uygulamayı başlat (Hugging Face 7860 portunu dinler, burası önemli!)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
