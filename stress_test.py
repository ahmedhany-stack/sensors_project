import asyncio
import random
import time
import httpx

URL = "http://127.0.0.1:8000/predict"

# 1. التوكن الخاص بالتوثيق
TOKEN = "fake-token"

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {TOKEN}"
}

async def send_single_request(client: httpx.AsyncClient, req_id: int):
    # ⚡ توليد بيانات عشوائية ومتغيرة في كل طلب عشان نكسر الـ Redis Cache
    random_cycles = random.randint(10, 300)
    random_setting_1 = round(random.uniform(-0.005, 0.005), 5)
    random_s2 = round(random.uniform(630.0, 650.0), 2)
    random_s7 = round(prev_s7 := random.uniform(540.0, 560.0), 2)

    payload = {
        "records": [
            {
                "unit_number": random.randint(1, 5),  # تغيير رقم الـ Unit عشوائي بين 1 و 5
                "time_in_cycles": random_cycles,    # تغيير الدورات عشوائياً
                "setting_1": random_setting_1,
                "setting_2": -0.0004,
                "setting_3": 100.0,
                "s_1": 518.67,
                "s_2": random_s2,
                "s_3": 1589.70,
                "s_4": 1400.60,
                "s_5": 14.62,
                "s_6": 21.61,
                "s_7": random_s7,
                "s_8": 2388.06,
                "s_9": 9046.19,
                "s_10": 1.30,
                "s_11": 47.47,
                "s_12": 521.66,
                "s_13": 2388.02,
                "s_14": 8138.62,
                "s_15": 8.4195,
                "s_16": 0.03,
                "s_17": 392,
                "s_18": 2388,
                "s_19": 100.0,
                "s_20": 39.06,
                "s_21": 23.4190
            }
        ]
    }

    start = time.perf_counter()
    try:
        response = await client.post(URL, json=payload, headers=HEADERS, timeout=90.0)
        elapsed = (time.perf_counter() - start) * 1000
        
        # طباعة أول خطأ لمعرفة السبب فوراً
        if response.status_code != 200 and req_id == 0:
            print(f"⚠️ تفاصيل خطأ أول طلب (Status {response.status_code}): {response.text}")

        return {
            "id": req_id,
            "status_code": response.status_code,
            "latency_ms": elapsed,
            "success": response.status_code == 200
        }
    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        if req_id == 0:
            print(f"⚠️ Exception في أول طلب: {e}")
        return {
            "id": req_id,
            "status_code": 500,
            "latency_ms": elapsed,
            "success": False,
            "error": str(e)
        }

async def run_stress_test(num_requests: int = 8000):
    print(f"💥 بدء إطلاق {num_requests} طلب دفعة واحدة (Extreme Stress Test)... ربنا يستر على اللابتوب!")
    
    # رفع حدود الـ Concurrency والـ Connections للحد الأقصى للـ OS
    semaphore = asyncio.Semaphore(250)
    limits = httpx.Limits(max_connections=10000, max_keepalive_connections=2000)
    
    async with httpx.AsyncClient(limits=limits, timeout=90.0) as client:
        overall_start = time.perf_counter()
        
        async def sem_task(j):
            async with semaphore:
                return await send_single_request(client, j)

        tasks = [sem_task(j) for j in range(num_requests)]
        results = await asyncio.gather(*tasks)
            
        overall_time = time.perf_counter() - overall_start

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    latencies = [r["latency_ms"] for r in results]

    status_codes = {}
    for r in results:
        status_codes[r["status_code"]] = status_codes.get(r["status_code"], 0) + 1

    print(f"\n📊 ================= نتائج اختبار الـ {num_requests} طلب =================")
    print(f"⏱️  الزمن الكلي:               {overall_time:.3f} ثانية")
    print(f"🚀  معدل الإنتاجية:           {num_requests / overall_time:.2f} Req/Sec")
    print(f"✅  الناجحة (200 OK):          {len(successful)} / {num_requests}")
    print(f"❌  الفاشلة:                   {len(failed)}")
    print(f"🔍  توزيع رموز الاستجابة:     {status_codes}")
    print(f"⚡  متوسط الـ Latency:         {sum(latencies)/len(latencies):.2f} ms")
    print("==================================================")

if __name__ == "__main__":
    # تقدر تغير الرقم هنا لـ 8000 أو 10000 براحتك
    asyncio.run(run_stress_test(12000))