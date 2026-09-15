import asyncio
import logging
from typing import List, Dict, Any
from src.api.database import save_predictions_bulk_to_db

logger = logging.getLogger("mlops_app")

class PredictionQueueWorker:
    def __init__(self, batch_size: int = 50, flush_interval: float = 2.0):
        self.queue = asyncio.Queue()
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self._worker_task = None
        self._is_running = False

    async def start(self):
        """تشغيل الـ Background Worker مع بدء السيرفر"""
        if self._is_running:
            return
        self._is_running = True
        self._worker_task = asyncio.create_task(self._process_queue())
        logger.info("Prediction Queue Worker started successfully.")

    async def stop(self):
        """إيقاف الـ Worker بأمان وتفريغ المتبقي في الـ Queue"""
        if not self._is_running:
            return
        self._is_running = False
        if self._worker_task:
            await self._flush_remaining()
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("Prediction Queue Worker stopped gracefully.")

    async def put(self, input_data: dict, prediction: float, unit_number: int, time_in_cycles: int):
        """إضافة نتيجة التنبؤ إلى الطابور فوراً بدون أي انتظار"""
        await self.queue.put({
            "input_data": input_data,
            "prediction": prediction,
            "unit_number": unit_number,
            "time_in_cycles": time_in_cycles
        })

    async def _process_queue(self):
        """حلقة العمل المستمرة لتجميع وحفظ الدفعات"""
        batch: List[Dict[Any, Any]] = []
        last_flush_time = asyncio.get_event_loop().time()

        while self._is_running:
            try:
                # حساب الـ Timeout المتبقي للوصول للوقت المخصص للـ Flush
                elapsed = asyncio.get_event_loop().time() - last_flush_time
                timeout = max(0.05, self.flush_interval - elapsed)

                try:
                    # انتظار سحب عنصر من الطابور أو انتهاء المهلة الزمنية
                    item = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                    batch.append(item)
                    self.queue.task_done()
                except asyncio.TimeoutError:
                    pass

                current_time = asyncio.get_event_loop().time()
                # تنفيذ الحفظ في حال اكتمل حجم الـ Batch أو انتهت المهلة الزمنية مع وجود عناصر
                if len(batch) >= self.batch_size or (batch and (current_time - last_flush_time) >= self.flush_interval):
                    await self._flush_batch(batch)
                    batch = []
                    last_flush_time = current_time

            except Exception as e:
                logger.error(f"Error in prediction queue worker loop: {e}", exc_info=True)
                await asyncio.sleep(0.5)

    async def _flush_batch(self, batch: List[Dict[Any, Any]]):
        """إرسال الدفعة لقاعدة البيانات في Thread منفصل لعدم تعطيل الـ Event Loop"""
        if not batch:
            return
        try:
            await asyncio.to_thread(save_predictions_bulk_to_db, batch)
            logger.debug(f"Successfully bulk inserted {len(batch)} predictions into DB.")
        except Exception as db_err:
            logger.error(f"Failed to bulk insert predictions into DB: {db_err}")

    async def _flush_remaining(self):
        """تفريغ أي عناصر بقيت في الطابور عند إغلاق التطبيق"""
        batch = []
        while not self.queue.empty():
            try:
                batch.append(self.queue.get_nowait())
                self.queue.task_done()
            except asyncio.QueueEmpty:
                break
        if batch:
            logger.info(f"Flushing remaining {len(batch)} items during shutdown...")
            await self._flush_batch(batch)

# إنشاء نسخة عامة (Singleton) للاستخدام في التطبيق
prediction_worker = PredictionQueueWorker(batch_size=50, flush_interval=2.0)