FROM apache/airflow:2.7.1

USER root

# إعادة توجيه مستودعات دبيان بولز القديمة إلى الأرشيف لتجنب أخطاء 404
RUN sed -i 's/deb.debian.org/archive.debian.org/g' /etc/apt/sources.list && \
    sed -i 's/security.debian.org/archive.debian.org/g' /etc/apt/sources.list && \
    sed -i '/buster-updates/d' /etc/apt/sources.list && \
    sed -i '/bullseye-updates/d' /etc/apt/sources.list

# تثبيت أدوات النظام المطلوبة بسلاسة
RUN apt-get update --allow-insecure-repositories || true && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# نسخ ملف المتطلبات وتثبيتها
COPY --chown=airflow:root requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# <--- الإضافة الهامة جداً: نسخ باقي ملفات المشروع (فولدر src وغيره) جوه مسار العمل بالحاوية --->
COPY --chown=airflow:root . /app
WORKDIR /app