import json
import redis.asyncio as aioredis
import numpy as np
from src.utils.logger import logger

class RedisFeatureStore:
    def __init__(self, host='localhost', port=6379, window_size=10):
        # استخدام النسخة غير المتزامنة من Redis
        self.client = aioredis.Redis(host=host, port=port, decode_responses=True)
        self.window_size = window_size

    async def get_rolling_and_lags_batch(self, records: list, sensor_cols: list):
        keys = [f"unit:{int(row.get('unit_number', 1))}:history" for row in records]
        
        # 🚀 استخدام await مع الـ MGET غير المتزامن
        histories = await self.client.mget(keys)
        
        processed_rows = []
        pipe_updates = []
        n_sensors = len(sensor_cols)

        for row, hist_json in zip(records, histories):
            current_vals = [float(row.get(col, 0.0)) for col in sensor_cols]
            
            history_matrix = []
            if hist_json:
                try:
                    raw_history = json.loads(hist_json)
                    for item in raw_history:
                        if isinstance(item, dict):
                            history_matrix.append([float(item.get(col, 0.0)) for col in sensor_cols])
                        elif isinstance(item, list):
                            history_matrix.append(item)
                except Exception:
                    history_matrix = []
            
            history_matrix.append(current_vals)
            if len(history_matrix) > self.window_size:
                history_matrix.pop(0)
                
            unit_num = int(row.get('unit_number', 1))
            pipe_updates.append((f"unit:{unit_num}:history", json.dumps(history_matrix)))
            
            arr = np.array(history_matrix, dtype=np.float32)
            
            if arr.shape[0] > 1:
                lags = arr[-2, :]
            else:
                lags = arr[-1, :]
                
            means = np.mean(arr, axis=0)
            stds = np.std(arr, axis=0) if arr.shape[0] > 1 else np.zeros(n_sensors, dtype=np.float32)
            
            enriched_row = row.copy()
            for idx, col in enumerate(sensor_cols):
                enriched_row[f'{col}_lag_1'] = float(lags[idx])
                enriched_row[f'{col}_rolling_mean'] = float(means[idx])
                enriched_row[f'{col}_rolling_std'] = float(stds[idx])

            processed_rows.append(enriched_row)
            
        # 🚀 استخدام await مع الـ Pipeline غير المتزامن
        async with self.client.pipeline(transaction=True) as pipe:
            for key, val in pipe_updates:
                pipe.set(key, val)
            await pipe.execute()
            
        return processed_rows