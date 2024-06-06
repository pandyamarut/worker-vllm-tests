import json
import runpod
import os
import asyncio
from tqdm import tqdm
from workers import vLLMWorker
from datetime import datetime
from dotenv import load_dotenv
from tests import OpenAITest
from time import sleep
import json


GPUS = ["ADA_24"]
MODELS = ["openchat/openchat-3.5-0106"]
IMAGE = "runpod/worker-vllm:dev-cuda12.1.0"
TESTNAME = "first_try"
CONTAINER_DISK_IN_GB = 200
MAX_INIT_WAIT = 10000 # seconds// 

load_dotenv()

# Retrieve the API key
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
# Set the API key for the runpod library
runpod.api_key = RUNPOD_API_KEY



def get_endpoint_health(endpoint_id):
    endpoint = runpod.Endpoint(endpoint_id)
    endpoint_health = endpoint.health()
    return endpoint_health


def if_endpoint_exists():
    endpoints = runpod.get_endpoints()
    for ep in endpoints:
        if "VLLMTEST" in ep['name']:
            print("You have already created an endpoint with the name:", ep['name'])

def test_results(results):
    total_tests = len(results)
    passed_tests = sum(1 for test in results.values() if test["status"] == "pass")
    pretty_results = json.dumps(results, indent=4)
    print(pretty_results)
    print(f"Total tests: {total_tests}, Passed tests: {passed_tests}")


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
                print(f"Created worker {worker.name}")
                endpoint_health = get_endpoint_health(worker.endpoint_id)
                 # polling the endpoint health  until it is healthy
                while not (endpoint_health["workers"]["initializing"] == 0 and endpoint_health["workers"]["running"] > 0):
                    print("Initializing the Worker")
                    sleep(5)  # wait for 5 seconds
                    endpoint_health = get_endpoint_health(worker.endpoint_id)
                print("Endpoint is healthy, Running Tests now.")
                try:
                    test = OpenAITest(
                    model_name=model,
                    base_url=worker.openai_base_url,
                    api_key=os.getenv("RUNPOD_API_KEY")
                )
                    results = await test.run_tests()
                except Exception as e:
                    print(f"Failed to run tests: {e}")
                    continue
                test_results(results)
                print("Tests are Completed, Deleting the worker....")
                worker.delete()
            except Exception as e:
                print(f"Seems Like test worker is already created, Delete the worker from the UI and Try Again")


if __name__ == "__main__":
    asyncio.run(main())