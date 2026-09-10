import os
import logging
from typing import Dict, Any, Optional
import pandas as pd
import yaml

# Imports المعتمدة والمستقرة لنسخة Evidently 0.4.5
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """دالة مساعدة لتحميل الإعدادات من ملف YAML."""
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


# تحميل الإعدادات
config = load_config()
monitoring_cfg = config.get("monitoring", {})

# إعداد الـ Logger بناءً على الـ Config
logger_name = monitoring_cfg.get("logger_name", "RUL_Monitoring")
logger = logging.getLogger(logger_name)


class EvidentlyDriftMonitor:
    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.reference_data: Optional[pd.DataFrame] = None
        # استخدام الإعدادات الممررة أو الجلب من الإعدادات العامة
        self.cfg = cfg or monitoring_cfg.get("drift", {})

    def set_reference_dataframe(self, df: pd.DataFrame):
        self.reference_data = df
        logger.info("Reference DataFrame configured successfully.")

    def run_drift_analysis(
        self, 
        current_data: pd.DataFrame, 
        save_html: Optional[bool] = None, 
        html_report_path: Optional[str] = None
    ) -> Dict[str, Any]:
        
        # استخراج القيم الافتراضية من الـ Config إن لم يتم إرسالها للدالة
        if save_html is None:
            save_html = self.cfg.get("default_save_html", True)
            
        if html_report_path is None:
            html_report_path = self.cfg.get("default_html_report_path", "reports/drift_report.html")

        if self.reference_data is None:
            logger.warning("No reference dataset set. Skipping drift calculation.")
            return {"status": "skipped"}

        try:
            # إنشاء التقرير
            report = Report(metrics=[
                DataDriftPreset(),
                DataQualityPreset()
            ])

            # تشغيل التحليل
            report.run(
                reference_data=self.reference_data,
                current_data=current_data
            )

            # حفظ ملف الـ HTML
            if save_html:
                os.makedirs(os.path.dirname(html_report_path), exist_ok=True)
                report.save_html(html_report_path)

            return {"status": "success", "report_path": html_report_path}

        except Exception as e:
            logger.error(f"Error executing Evidently drift analysis: {str(e)}")
            return {"status": "error", "message": str(e)}


drift_monitor = EvidentlyDriftMonitor()