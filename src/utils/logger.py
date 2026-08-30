import logging
import os
from datetime import datetime

# 1. إنشاء فولدر الـ logs
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# 2. اسم ملف الـ log باليوم والساعة
LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# 3. ضبط إعدادات الـ logging (تطبع في الكونسول وتتحفظ في ملف)
logging.basicConfig(
    level=logging.INFO,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH), # حفظ في ملف
        logging.StreamHandler()             # طباعة على الشاشة
    ]
)

# تصدير الـ logger جاهز للاستخدام
logger = logging.getLogger("RUL_Logger")