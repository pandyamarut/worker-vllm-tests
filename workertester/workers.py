from typing import Dict, Any
import runpod
from config import vLLMWorkerConfig
from datetime import datetime
import requests

class RunPodTemplate:
    def __init__(self, name: str, image_name: str, env: Dict[str, str], container_disk_in_gb: int):
        self.name = name
        self.id = runpod.create_template(
            name=name,
            image_name=image_name,
            env=env,
            is_serverless=True,
            container_disk_in_gb=container_disk_in_gb,
        )["id"]

    def delete(self):
        runpod.delete_template(self.name)
        


class vLLMWorker:
    def __init__(self, name, image_name, env, gpu_ids, container_disk_in_gb, api_key, url_prefix="https://api.runpod.ai/v2"):
        self.name = name
        self.image_name = image_name
        self.env = env
        self.gpu_ids = gpu_ids
        self.container_disk_in_gb = container_disk_in_gb
        
        self.api_key = api_key
        runpod.api_key = api_key
        
        self.url_prefix = url_prefix
        
        self.openai_base_url = None
        self.endpoint_id = None
        self.template = None
        
    async def check_health(self):
        return self.runpod_endpoint.health() == {"status": "HEALTHY"}
        
    def create(self):
        try:
            self.template = RunPodTemplate(
                name=self.name,
                image_name=self.image_name,
                env=self.env,
                container_disk_in_gb=self.container_disk_in_gb,
            )
            print(f"Created template {self.name}")
        except Exception as e:
            if "must be unique" in str(e) or "Please try again" in str(e):
                try:
                    runpod.delete_template(self.name)
                except Exception as delete_e:
                    if "associated" in str(delete_e):
                        runpod.delete_endpoint(str(delete_e).split()[-1])
                        print(f"Deleted endpoint {str(delete_e).split()[-1]}")
                    else:
                        print(f"Failed to delete template: {delete_e}")
                finally:
                    self.template = RunPodTemplate(
                        name=self.name,
                        image_name=self.image_name,
                        env=self.env,
                        container_disk_in_gb=self.container_disk_in_gb,
                    )
            elif "associated" in str(e):
                print(f"Template {self.name} is associated with an endpoint. Deleting endpoint.")
                runpod.delete_endpoint(str(e).split()[-1])
                self.endpoint_id = None
                self.template = RunPodTemplate(
                    name=self.name,
                    image_name=self.image_name,
                    env=self.env,
                    container_disk_in_gb=self.container_disk_in_gb,
                )
            else:
                print(f"Unhandled exception: {e}")
                raise e

        self.endpoint_id = runpod.create_endpoint(**vLLMWorkerConfig(
            name=self.name,
            template_id=self.template.id,
            gpu_ids=self.gpu_ids,
        ).model_dump())["id"]
        self.runpod_endpoint = runpod.Endpoint(self.endpoint_id)
        self.openai_base_url = f"{self.url_prefix}/{self.endpoint_id}/openai/v1"

        

    # Delete the endpoint using the GraphQL API (TODO: Add a method to delete the endpoint using the SDK)
    def delete(self):
        url = f"https://api.runpod.io/graphql?api_key={self.api_key}"
        headers = {
            'content-type': 'application/json'
        }
        data = {
            "query": f"mutation {{ deleteEndpoint(id: \"{self.endpoint_id}\") }}"
        }
        try:
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                print("Endpoint deleted successfully.")
            else:
                print("Failed to delete endpoint.")
        except Exception as e:
            print(f"Failed to delete endpoint: {e}")
        

