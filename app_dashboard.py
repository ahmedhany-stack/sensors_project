import os
import requests
import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine

# إعدادات الصفحة
st.set_page_config(
    page_title="RUL Prediction & MLOps Dashboard",
    page_icon="✈️",
    layout="wide",
)

# ------------------------------------------------------------------------------
# Configuration & Connections
# ------------------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://airflow:airflow@127.0.0.1:5433/rul_db")
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")

@st.cache_resource
def get_engine():
    return create_engine(DATABASE_URL)

engine = get_engine()

# دالة لجلب الميتريكس مباشرة من Prometheus
def query_prometheus(promql_query):
    try:
        response = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={"query": promql_query}, timeout=3)
        data = response.json()
        results = data.get("data", {}).get("result", [])
        if results:
            return float(results[0]["value"][1])
        return 0.0
    except Exception:
        return 0.0

# ------------------------------------------------------------------------------
# Dashboard Header
# ------------------------------------------------------------------------------
st.title("⚙️ Engine RUL Monitoring & MLOps Dashboard")
st.markdown("لوحة متابعة عمر المحركات المتبقي (RUL) ومراقبة أداء الـ API لحظة بلحظة")

if st.button("🔄 تحديث البيانات"):
    st.rerun()

# ------------------------------------------------------------------------------
# القسم الأول: Prometheus Real-Time Metrics (الأداء والصيانة)
# ------------------------------------------------------------------------------
st.subheader("📊 إحصائيات الـ API والحمولة (Prometheus Real-Time)")

# جلب البيانات من Prometheus
total_success = query_prometheus('app_requests_total{endpoint="/predict", status_code="200"}')
total_maintenance = query_prometheus('app_requests_total{endpoint="/predict", status_code="503"}')
cache_hits = query_prometheus('cache_hits_total{status="hit"}')
cache_misses = query_prometheus('cache_hits_total{status="miss"}')

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric(label="الطلبات الناجحة (200 OK)", value=int(total_success))
with m2:
    st.metric(label="طلبات مرفوضة - صيانة (503)", value=int(total_maintenance), delta_color="inverse")
with m3:
    st.metric(label="Cache Hits (Redis)", value=int(cache_hits))
with m4:
    total_reqs = cache_hits + cache_misses
    hit_ratio = round((cache_hits / total_reqs * 100), 1) if total_reqs > 0 else 0.0
    st.metric(label="نسبة الكاش (Cache Hit Ratio)", value=f"{hit_ratio}%")

st.markdown("---")

# ------------------------------------------------------------------------------
# القسم الثاني: بيانات التنبؤات المخزنة في قاعدة البيانات (Database Logs)
# ------------------------------------------------------------------------------
try:
    query = "SELECT * FROM prediction_logs ORDER BY created_at DESC;"
    df = pd.read_sql(query, con=engine)
except Exception as e:
    st.error(f"فشل الاتصال بقاعدة البيانات: {e}")
    df = pd.DataFrame()

if df.empty:
    st.warning("⚠️ لا توجد سجلات تنبؤات ناجحة في قاعدة البيانات حتى الآن.")
else:
    # كروت إحصائية للـ DB
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="إجمالي التنبؤات المسجلة بالداتابيز", value=len(df))
    with col2:
        unique_units = df["unit_number"].nunique() if "unit_number" in df else 0
        st.metric(label="عدد المحركات الفريدة", value=unique_units)
    with col3:
        avg_rul = round(df["predicted_rul"].mean(), 2) if "predicted_rul" in df else 0
        st.metric(label="متوسط الـ RUL المتوقع", value=f"{avg_rul} cycles")

    st.markdown("---")

    # قسم الرسوم البيانية والتنبيهات
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("📈 تدهور العمر المتوقع (RUL Degradation Over Time)")
        if "unit_number" in df and "predicted_rul" in df and "created_at" in df:
            fig = px.line(
                df.sort_values("created_at"),
                x="created_at",
                y="predicted_rul",
                color="unit_number",
                markers=True,
                labels={
                    "created_at": "وقت الطلب",
                    "predicted_rul": "العمر المتبقي (RUL)",
                    "unit_number": "رقم المحرك",
                },
                title="تطور الـ RUL للمحركات عبر الزمن",
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("🚨 محركات تحت الخط الحرج (Alerts)")
        threshold = 30
        if "predicted_rul" in df:
            critical_units = df[df["predicted_rul"] < threshold]
            if not critical_units.empty:
                st.error(f"تحذير: يوجد {len(critical_units)} سجل بحالة حرجة!")
                st.dataframe(
                    critical_units[["unit_number", "time_in_cycles", "predicted_rul"]]
                )
            else:
                st.success("✅ جميع المحركات تعمل بشكل سليم ولا توجد تنبيهات حرجة حالياً.")

    st.markdown("---")
    st.subheader("📋 سجلات التنبؤات الكاملة (Prediction Logs)")
    st.dataframe(df, use_container_width=True)