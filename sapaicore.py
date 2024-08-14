"""
title: Open WebUI - SAP API Core Pipeline
author: schardosin
date: 2024-08-12
version: 1.0
license: MIT
description: A pipeline for generating text using the SAP API Core.
requirements: requests, sseclient-py
"""

from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
import os
import requests
import sseclient
import json

class Pipeline:
    class Valves(BaseModel):
        AI_CORE_CLIENT_ID: str = ""
        AI_CORE_CLIENT_SECRET: str = ""
        AI_CORE_TOKEN_URL: str = ""
        AI_CORE_BASE_URL: str = ""

    def __init__(self):
        self.type = "manifold"
        self.id = "sapaicore"
        self.name = "sapaicore/"
        self.pipelines = []
        self.valves = self.Valves(
            **{
                "AI_CORE_CLIENT_ID": os.getenv("AI_CORE_CLIENT_ID", ""),
                "AI_CORE_CLIENT_SECRET": os.getenv("AI_CORE_CLIENT_SECRET", ""),
                "AI_CORE_TOKEN_URL": os.getenv("AI_CORE_TOKEN_URL", "https://<account>.authentication.sap.hana.ondemand.com/oauth/token"),
                "AI_CORE_BASE_URL": os.getenv("AI_CORE_BASE_URL", "https://api.ai.<region>.aws.ml.hana.ondemand.com/v2")
            }
        )

    async def on_startup(self):
        print(f"on_startup:{__name__}")

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")

    async def on_valves_updated(self):
        self.update_pipelines()
        print(f"on_valves_updated{__name__}")

    def get_aicore_deployments(self):
        if self.valves.AI_CORE_CLIENT_SECRET == "":
            return [
                {"id": "notconfigured", "name": "ai-core-not-configured"}
            ]
        else:
            token = self.authenticate()
            headers = {
                'Authorization': f'Bearer {token}',
                'AI-Resource-Group': 'default',
                'Content-Type': 'application/json',
            }
            url = f"{self.valves.AI_CORE_BASE_URL}/lm/deployments?$top=10000&$skip=0"
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            deployments = response.json()['resources']
            
            # Filter running deployments and map data
            return [
                {
                    "id": deployment["id"],
                    "name": f"{deployment['details']['resources']['backend_details']['model']['name']}:{deployment['details']['resources']['backend_details']['model']['version']}"
                }
                for deployment in deployments if deployment["targetStatus"] == "RUNNING"
            ]
        
    def update_pipelines(self) -> None:
        self.pipelines = self.get_aicore_deployments()

    def pipelines(self) -> List[dict]:
        return self.get_aicore_deployments()
    
    def get_model_name(self, model_id: str) -> str:
        for pipeline in self.pipelines:
            if pipeline['id'] == model_id:
                return pipeline['name']
        return ""

    def authenticate(self):
        payload = {
            'grant_type': 'client_credentials',
            'client_id': self.valves.AI_CORE_CLIENT_ID,
            'client_secret': self.valves.AI_CORE_CLIENT_SECRET
        }
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        response = requests.post(self.valves.AI_CORE_TOKEN_URL, data=payload, headers=headers)
        response.raise_for_status()
        return response.json()['access_token']

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        print(f"pipe:{__name__}")

        model_name = self.get_model_name(model_id)

        token = self.authenticate()
        headers = {
            'Authorization': f'Bearer {token}',
            'AI-Resource-Group': 'default',
            'Content-Type': 'application/json',
        }

        complete_messages = [
            {
                "role": msg["role"], 
                "content": [{"type": "text", "text": msg["content"]}]
            } 
            for msg in messages
        ]
        #complete_messages.append({"role": "user", "content": [{"type": "text", "text": user_message}]})

        payload = {}
        if model_name.lower().startswith('anthropic'):
            payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "messages": complete_messages,
                "max_tokens": 4096,
                "temperature": 0.7
            }

            if body.get("stream", False):
               return self.stream_response_claude(payload, headers, model_id)
            else:
                return self.get_completion_claude(payload, headers, model_id)

        else:             
            payload = {
                "messages": complete_messages,
                "max_tokens": 4096,
                "temperature": 0.7,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "stop": "null",
                "stream": body.get("stream", False),
            }

            if body.get("stream", False):
                return self.stream_response_gpt(payload, headers, model_id)
            else:
                return self.get_completion_gpt(payload, headers, model_id)
        
    def stream_response_gpt(self, payload: dict, headers: dict, model_id: str) -> Generator:
        url = f"{self.valves.AI_CORE_BASE_URL}/inference/deployments/{model_id}/chat/completions?api-version=2023-05-15"

        response = requests.post(url, headers=headers, json=payload, stream=True)
        if response.status_code == 200:
            client = sseclient.SSEClient(response)
            for event in client.events():
                try:
                    data = json.loads(event.data)
                    if 'choices' in data and len(data['choices']) > 0:
                        delta = data['choices'][0].get('delta', {})
                        if 'content' in delta:
                            yield delta['content']
                except json.JSONDecodeError:
                    print(f"Failed to parse JSON: {event.data}")
                except KeyError as e:
                    print(f"Unexpected data structure: {e}")
                    print(f"Full data: {data}")
        else:
            raise Exception(f"Error: {response.status_code} - {response.text}")

    def get_completion_gpt(self, payload: dict, headers: dict, model_id: str) -> str:
        url=f"{self.valves.AI_CORE_BASE_URL}/inference/deployments/{model_id}/chat/completions?api-version=2023-05-15"
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Error: {response.status_code} - {response.text}")

    def stream_response_claude(self, payload: dict, headers: dict, model_id: str) -> Generator:
        url = f"{self.valves.AI_CORE_BASE_URL}/inference/deployments/{model_id}/invoke-with-response-stream"

        response = requests.post(url, headers=headers, json=payload, stream=True)
        if response.status_code == 200:
            client = sseclient.SSEClient(response)
            for event in client.events():
                try:
                    data = json.loads(event.data)
                    
                    # Check for content_block_delta to yield text
                    if 'delta' in data and data['delta'].get('type') == 'text_delta':
                        text_delta = data['delta'].get('text', '')
                        yield text_delta

                except json.JSONDecodeError:
                    print(f"Failed to parse JSON: {event.data}")
                except KeyError as e:
                    print(f"Unexpected data structure: {e}")
                    print(f"Full data: {data}")
        else:
            raise Exception(f"Error: {response.status_code} - {response.text}")


    def get_completion_claude(self, payload: dict, headers: dict, model_id: str) -> str:
        url = f"{self.valves.AI_CORE_BASE_URL}/inference/deployments/{model_id}/invoke"
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            res = response.json()
            return res["content"][0]["text"] if "content" in res and res["content"] else ""
        else:
            raise Exception(f"Error: {response.status_code} - {response.text}")