import os
import logging
from typing import Dict, Any, Optional
import pandas as pd

# Imports المعتمدة والمستقرة لنسخة Evidently 0.4.5
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset



logger = logging.getLogger("RUL_Monitoring")


class EvidentlyDriftMonitor:
    def __init__(self):
        self.reference_data: Optional[pd.DataFrame] = None

    def set_reference_dataframe(self, df: pd.DataFrame):
        self.reference_data = df
        logger.info("Reference DataFrame configured successfully.")

    def run_drift_analysis(
        self, 
        current_data: pd.DataFrame, 
        save_html: bool = True, 
        html_report_path: str = "reports/drift_report.html"
    ) -> Dict[str, Any]:
        
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

            

            return {"status": "success"}

        except Exception as e:
            logger.error(f"Error executing Evidently drift analysis: {str(e)}")
            return {"status": "error", "message": str(e)}


drift_monitor = EvidentlyDriftMonitor()