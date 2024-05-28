from workers import vLLMWorker
from datetime import datetime
from dotenv import load_dotenv
from tests import OpenAITest
import os
import asyncio
from time import sleep
import json


GPUS = ["ADA_24"]
MODELS = ["openchat/openchat-3.5-0106"]
IMAGE = "runpod/worker-vllm:dev-cuda12.1.0"
TESTNAME = "first_try"
CONTAINER_DISK_IN_GB = 200
MAX_INIT_WAIT = 10000 # seconds

async def main():
    load_dotenv()
    for gpu in GPUS:
        for model in MODELS:
            print(f"Running tests for {model} on {gpu}")
            worker = vLLMWorker(
                name=f"VLLMTEST-{TESTNAME}-{model.replace('/', '-').replace('.', '-')}-{gpu}",
                image_name=IMAGE,
                env={
                    "MODEL_NAME": model,
                    "MAX_MODEL_LEN": "1024",
                },
                gpu_ids=gpu,
                container_disk_in_gb=CONTAINER_DISK_IN_GB,
                api_key=os.getenv("RUNPOD_API_KEY"),
            )
            
            try:
                worker.create()
            except Exception as e:
                print(f"Failed to create worker: {e}")
                worker.delete()
                continue
            
            while not await worker.check_health() and MAX_INIT_WAIT >= 0:
                print(f"Waiting for {worker.name} to become healthy...")
                sleep(5)
                MAX_INIT_WAIT -= 5
                
            if MAX_INIT_WAIT <= 0:
                print(f"Worker {worker.name} did not become healthy in time. Exiting.")
                worker.delete()
                continue
            else:
                print(f"{worker.name} is healthy!")
                test = OpenAITest(
                    model_name=model,
                    base_url=worker.openai_base_url,
                    api_key=os.getenv("RUNPOD_API_KEY")
                )
                results = await test.run_tests()
                print(results)
                # Save to json
                with open(f"results/{TESTNAME}-{model.replace('/', '-').replace('.', '-')}-{gpu}.json", "w") as f:
                    json.dump(results, f)
            worker.delete()
            
            
            
if __name__ == "__main__":
    asyncio.run(main())