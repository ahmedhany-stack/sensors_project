import asyncio
import time
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DynamicBatcher:
    def __init__(self, pipeline, max_batch_size: int = 32, max_latency_ms: float = 5.0):
        self.pipeline = pipeline
        self.max_batch_size = max_batch_size
        self.max_latency_sec = max_latency_ms / 1000.0
        self.queue = asyncio.Queue()
        self.worker_task = None

    async def start(self):
        self.worker_task = asyncio.create_task(self._process_batches())

    async def stop(self):
        if self.worker_task:
            self.worker_task.cancel()

    async def predict(self, features_df: pd.DataFrame) -> list:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self.queue.put((features_df, future))
        return await future

    async def _process_batches(self):
        while True:
            batch = []
            start_time = time.time()

            first_item = await self.queue.get()
            batch.append(first_item)
            self.queue.task_done()

            while len(batch) < self.max_batch_size:
                elapsed = time.time() - start_time
                remaining_time = self.max_latency_sec - elapsed
                
                if remaining_time <= 0:
                    break

                try:
                    item = await asyncio.wait_for(self.queue.get(), timeout=remaining_time)
                    batch.append(item)
                    self.queue.task_done()
                except asyncio.TimeoutError:
                    break

            try:
                dfs = [item[0] for item in batch]
                futures = [item[1] for item in batch]

                # 1. دمج البيانات في Matrix واحدة
                combined_df = pd.concat(dfs, ignore_index=True)
                
                # 2. التنبؤ بالموديل
                raw_predictions = self.pipeline.predict(combined_df)

                # ⚡ 3. معالجة مخرجات ONNX/MLflow بجميع الأشكال الممكنة
                if isinstance(raw_predictions, tuple) or isinstance(raw_predictions, list):
                    # لو ONNX رجع list of outputs
                    raw_predictions = raw_predictions[0]

                if hasattr(raw_predictions, 'to_numpy'):
                    raw_preds_array = raw_predictions.to_numpy()
                elif isinstance(raw_predictions, pd.DataFrame) or isinstance(raw_predictions, pd.Series):
                    raw_preds_array = raw_predictions.values
                else:
                    raw_preds_array = np.array(raw_predictions)

                # تحويل أخير لـ Flat List أرقام
                raw_preds_list = raw_preds_array.astype(float).flatten().tolist()

                # 4. توزيع النتائج بحسب عدد صفوف كل Request
                curr_idx = 0
                for df, future in zip(dfs, futures):
                    req_len = len(df)
                    req_preds = raw_preds_list[curr_idx : curr_idx + req_len]
                    curr_idx += req_len
                    
                    if not future.done():
                        future.set_result(req_preds)

            except Exception as e:
                logger.error(f"Batcher Exception: {str(e)}", exc_info=True)
                for _, future in batch:
                    if not future.done():
                        future.set_exception(e)