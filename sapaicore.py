"""
title: Open WebUI - SAP API Core Pipeline with GenAI SDK and LangChain
author: schardosin
date: 2024-08-12
version: 1.1
license: MIT
description: A pipeline for generating text using the SAP API Core with GenAI SDK and LangChain.
requirements: generative-ai-hub-sdk[all], pydantic==2.7.4
"""

import os
from pydantic import BaseModel, Field
from typing import List, Union, Generator, Iterator
from gen_ai_hub.proxy.langchain.init_models import init_llm
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
from langchain.prompts import ChatPromptTemplate
from langchain.schema import AIMessage, HumanMessage

class Pipeline:
    class Valves(BaseModel):
        AI_CORE_CLIENT_ID: str = Field(default="")
        AI_CORE_CLIENT_SECRET: str = Field(default="")
        AI_CORE_AUTH_URL: str = Field(default="")
        AI_CORE_BASE_URL: str = Field(default="")
        AI_CORE_RESOURCE_GROUP: str = Field(default="")
        
    def __init__(self):
        self.type: str = "manifold"
        self.id: str = "sapaicore"
        self.name: str = "sapaicore/"
        self.pipelines: List[dict] = []
        self.valves = self.Valves(
            AI_CORE_CLIENT_ID=os.getenv("AI_CORE_CLIENT_ID", ""),
            AI_CORE_CLIENT_SECRET=os.getenv("AI_CORE_CLIENT_SECRET", ""),
            AI_CORE_AUTH_URL=os.getenv("AI_CORE_AUTH_URL", "https://<account>.authentication.sap.hana.ondemand.com"),
            AI_CORE_BASE_URL=os.getenv("AI_CORE_BASE_URL", "https://api.ai.<region>.aws.ml.hana.ondemand.com/v2"),
            AI_CORE_RESOURCE_GROUP = os.getenv("AICORE_RESOURCE_GROUP", "default")
        )
        self.proxy_client = None
        self.initialize_gen_ai_hub()

    def initialize_gen_ai_hub(self):
        os.environ["AICORE_AUTH_URL"] = self.valves.AI_CORE_AUTH_URL
        os.environ["AICORE_CLIENT_ID"] = self.valves.AI_CORE_CLIENT_ID
        os.environ["AICORE_CLIENT_SECRET"] = self.valves.AI_CORE_CLIENT_SECRET
        os.environ["AICORE_RESOURCE_GROUP"] = self.valves.AI_CORE_RESOURCE_GROUP
        os.environ["AICORE_BASE_URL"] = self.valves.AI_CORE_BASE_URL

        self.proxy_client = get_proxy_client("gen-ai-hub")
        self.proxy_client.refresh_instance_cache()

    async def on_startup(self):
        print(f"on_startup:{__name__}")

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")

    async def on_valves_updated(self):
        self.update_pipelines()
        print(f"on_valves_updated{__name__}")

    def get_aicore_deployments(self):
        if not self.proxy_client:
            return [{"id": "notconfigured", "name": "ai-core-not-configured"}]
        
        deployments = self.proxy_client.deployments
        return [
            {
                "id": deployment.model_name,
                "model": deployment.model_name,
                "name": f"{deployment.model_name}:{deployment.additonal_parameters.get('model_version', 'latest')}"
            }
            for deployment in deployments
        ]

    def update_pipelines(self) -> None:
        os.environ["AICORE_AUTH_URL"] = self.valves.AI_CORE_AUTH_URL
        os.environ["AICORE_CLIENT_ID"] = self.valves.AI_CORE_CLIENT_ID
        os.environ["AICORE_CLIENT_SECRET"] = self.valves.AI_CORE_CLIENT_SECRET
        os.environ["AICORE_RESOURCE_GROUP"] = "default"
        os.environ["AICORE_BASE_URL"] = self.valves.AI_CORE_BASE_URL

        self.proxy_client = get_proxy_client("gen-ai-hub")
        self.proxy_client.refresh_instance_cache()
        self.pipelines = self.get_aicore_deployments()

    def pipelines(self) -> List[dict]:
        return self.get_aicore_deployments()

    def get_model_name(self, model_id: str) -> str:
        for pipeline in self.pipelines:
            if pipeline['id'] == model_id:
                return pipeline['model']
        return ""

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        print(f"pipe:{__name__}")        

        llm = init_llm(model_id, proxy_client=self.proxy_client, streaming=body.get("stream", False), max_tokens=2048)

        chat_messages = []
        for msg in messages:
            if msg["role"] == "user":
                chat_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                chat_messages.append(AIMessage(content=msg["content"]))
        
        chat_prompt = ChatPromptTemplate.from_messages(chat_messages)

        if body.get("stream", False):
            return self.stream_response(llm, chat_prompt)
        else:
            return self.get_completion(llm, chat_prompt)

    def stream_response(self, llm, chat_prompt):
        try:
            for chunk in llm.stream(chat_prompt.format_prompt().to_messages()):
                yield chunk.content
        except Exception as e:
            if "streaming" in str(e) and "false" in str(e):
                response = self.get_completion(llm, chat_prompt)
                yield response
            else:
                raise

    def get_completion(self, llm, chat_prompt):
        reply = llm.invoke(chat_prompt.format_prompt().to_messages())
        return reply.content
