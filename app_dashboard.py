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

# الاتصال بقاعدة البيانات (نفس الرابط اللي شغال معاك)
DATABASE_URL = "postgresql+psycopg2://airflow:airflow@127.0.0.1:5433/rul_db"


@st.cache_resource
def get_engine():
  return create_engine(DATABASE_URL)


engine = get_engine()

# عنوان الداشبورد
st.title("⚙️ Engine RUL Monitoring & Prediction Dashboard")
st.markdown(
    "لوحة متابعة عمر الآلات والمحركات المتبقي (Remaining Useful Life) لحظة بلحظة"
)

# زرار لتحديث البيانات
if st.button("🔄 تحديث البيانات"):
  st.rerun()

# جلب البيانات من جدول prediction_logs
try:
  query = "SELECT * FROM prediction_logs ORDER BY created_at DESC;"
  df = pd.read_sql(query, con=engine)
except Exception as e:
  st.error(f"فشل الاتصال بقاعدة البيانات: {e}")
  df = pd.DataFrame()

# التحقق مما إذا كانت الداتابيز فارغة أم لا
if df.empty:
  st.warning(
      "⚠️ لا توجد بيانات مسجلة حتى الآن. ابعت بعض الطلبات (Requests) للـ FastAPI"
      " أولاً!"
  )
else:
  # كروت إحصائية سريعة في الأعلى
  col1, col2, col3 = st.columns(3)
  with col1:
    st.metric(
        label="إجمالي التنبؤات المسجلة",
        value=len(df),
    )
  with col2:
    unique_units = df["unit_number"].nunique() if "unit_number" in df else 0
    st.metric(label="عدد المحركات الفريدة", value=unique_units)
  with col3:
    avg_rul = round(df["predicted_rul"].mean(), 2) if "predicted_rul" in df else 0
    st.metric(label="متوسط الـ RUL المتوقع", value=f"{avg_rul} cycles")

  st.markdown("---")

  # قسم الرسوم البيانية والجداول
  col_left, col_right = st.columns([2, 1])

  with col_left:
    st.subheader("📈 تدهور العمر المتوقع (RUL Degradation Over Time)")
    if "unit_number" in df and "predicted_rul" in df:
      # رسم بياني تفاعلي يوضح التنبؤات لكل محرك
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
    st.subheader("🚨 محركات تحت الخط الحرطر (Alerts)")
    # افتراض أن الخط الحرطر أقل من 30 دورة
    threshold = 30
    if "predicted_rul" in df:
      critical_units = df[df["predicted_rul"] < threshold]
      if not critical_units.empty:
        st.error(f"تحذير: يوجد {len(critical_units)} سجل بحالة حرجة!")
        st.dataframe(
            critical_units[["unit_number", "time_in_cycles", "predicted_rul"]]
        )
      else:
        st.success(
            "✅ جميع المحركات تعمل بشكل سليم ولا توجد تنبيهات حرجة حالياً."
        )

  st.markdown("---")
  # عرض الجدول الكامل للـ Logs
  st.subheader("📋 سجلات التنبؤات الكاملة (Prediction Logs)")
  st.dataframe(df, use_container_width=True)