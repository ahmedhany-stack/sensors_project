def main():
    print("Hello from sensors-project!")


if __name__ == "__main__":
    main()



"""

FROM python:3.11-slim

# تحديد مجلد العمل داخل الـ Container
WORKDIR /app

# لو محتاج Curl أو حزم نظام أساسية مستقبلاً (من غير build-essential التقيل)
# لو مش محتاج أي حزم من apt تقدر تشيل RUN apt-get خالص!
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# نسخ ملف الـ Requirements أولاً للاستفادة من الـ Caching
COPY requirements.txt .

# تحديث pip وتثبيت المكتبات
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# نسخ باقي كود المشروع
COPY . .

# فتح البورت الخاص بالـ API
EXPOSE 8000

# أمر التشغيل عبر Uvicorn
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
      
      
      
      
      
      
      """