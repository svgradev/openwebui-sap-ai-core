"""
title: SAP AI Core
author: sgradev
date: 2025-07-07
version: 7.5 (Stable)
license: MIT
description: A pipe for SAP AI Core provider.
requirements: requests
"""

import os
import json
import requests
import ast  # Added for safely parsing non-standard JSON from Anthropic
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel, Field
from typing import List, Union, Generator, Dict

# --- Main Pipe Class ---


class Pipe:
    """
    Connects to SAP AI Core using direct REST API calls. This implementation
    avoids the SAP GenAI Hub SDK and LangChain for greater control and stability.
    """

    class Valves(BaseModel):
        """Configuration settings managed in the Open WebUI 'Pipes' panel."""

        AI_CORE_CLIENT_ID: str = Field(default="", title="SAP AI Core Client ID")
        AI_CORE_CLIENT_SECRET: str = Field(
            default="", title="SAP AI Core Client Secret"
        )
        AI_CORE_AUTH_URL: str = Field(
            default="",
            title="SAP AI Core Auth URL (e.g., https://<subdomain>.authentication.sap.hana.ondemand.com)",
        )
        AI_CORE_BASE_URL: str = Field(
            default="",
            title="SAP AI Core Base URL (e.g., https://api.ai.prod.eu-central-1.aws.ml.hana.ondemand.com)",
        )
        AI_CORE_RESOURCE_GROUP: str = Field(
            default="default", title="SAP AI Core Resource Group"
        )
        MAX_TOKENS: int = Field(default=8192, title="Max Tokens")
        TEMPERATURE: float = Field(default=0.7, title="Default Temperature")
        STREAM: bool = Field(default=True, title="Enable Streaming")

    def __init__(self):
        """Initializes the pipe state. Token and deployments are initially None."""
        self.valves = self.Valves()
        self.token_info: Dict = {}
        self.deployments: List[Dict] = []

    # --- Authentication ---

    def _get_token(self) -> str:
        """
        Retrieves a valid OAuth token, authenticating if necessary.
        Caches the token until it expires.
        """
        # Check if token exists and is not expired (with a 60-second buffer)
        if (
            self.token_info
            and self.token_info.get("expires_at", 0)
            > (datetime.now(timezone.utc) + timedelta(seconds=60)).timestamp()
        ):
            return self.token_info["access_token"]

        print("Authenticating with SAP AI Core...")

        token_url = f"{self.valves.AI_CORE_AUTH_URL.rstrip('/')}/oauth/token"
        payload = {
            "grant_type": "client_credentials",
            "client_id": self.valves.AI_CORE_CLIENT_ID,
            "client_secret": self.valves.AI_CORE_CLIENT_SECRET,
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        try:
            response = requests.post(token_url, data=payload, headers=headers)
            response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)

            data = response.json()
            expires_at = datetime.now(timezone.utc).timestamp() + data["expires_in"]
            self.token_info = {
                "access_token": data["access_token"],
                "expires_at": expires_at,
            }
            print("✅ Authentication successful.")
            return self.token_info["access_token"]
        except requests.exceptions.RequestException as e:
            print(f"❌ Authentication failed: {e}")
            raise

    # --- Deployment Management ---

    def _get_deployments(self) -> List[Dict]:
        """Fetches and caches the list of running deployments from SAP AI Core."""
        # Return cached deployments if available
        if self.deployments:
            return self.deployments

        print("Fetching deployments from SAP AI Core...")
        token = self._get_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "AI-Resource-Group": self.valves.AI_CORE_RESOURCE_GROUP,
        }
        # Using the /v2/lm/deployments endpoint as it appears to be the correct one for listing foundation model deployments.
        url = f"{self.valves.AI_CORE_BASE_URL.rstrip('/')}/v2/lm/deployments?$top=1000"

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            api_deployments = response.json().get("resources", [])

            # Filter for running deployments and format them based on the /lm/deployments structure
            formatted_deployments = []
            for dep in api_deployments:
                if dep.get("targetStatus") == "RUNNING":
                    try:
                        # Safely navigate the nested dictionary structure from the TS example
                        model_name = dep["details"]["resources"]["backend_details"][
                            "model"
                        ]["name"]
                        deployment_id = dep["id"]
                        # The internal cache stores the real deployment ID and the model name
                        formatted_deployments.append(
                            {"id": deployment_id, "name": model_name}
                        )
                    except (KeyError, TypeError):
                        # This handles cases where the nested keys don't exist
                        print(
                            f"Warning: Skipping deployment {dep.get('id')} due to unexpected data structure."
                        )
                        continue

            self.deployments = formatted_deployments
            print(f"✅ Found {len(self.deployments)} running deployments.")
            return self.deployments
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to fetch deployments: {e}")
            raise

    def _get_deployment_id_for_model(self, model_name: str) -> str:
        """Finds the deployment ID for a given model name."""
        deployments = self._get_deployments()
        for dep in deployments:
            # Match model name exactly
            if dep["name"] == model_name:
                return dep["id"]
        raise ValueError(f"No running deployment found for model '{model_name}'")

    # --- Main Pipe Methods ---

    def pipes(self) -> List[dict]:
        """
        Called by Open WebUI to discover available models.
        It returns a list where the model's name is used as the unique ID for the UI.
        """
        if not all(
            [
                self.valves.AI_CORE_CLIENT_ID,
                self.valves.AI_CORE_CLIENT_SECRET,
                self.valves.AI_CORE_AUTH_URL,
                self.valves.AI_CORE_BASE_URL,
            ]
        ):
            return [
                {
                    "id": "not-configured",
                    "name": "Pipe is not configured. Please check settings.",
                }
            ]
        try:
            # This populates self.deployments cache for internal use
            all_deployments = self._get_deployments()

            # Use a dictionary to handle cases where multiple deployments might have the same model name,
            # ensuring the UI list has unique entries.
            unique_models_for_ui = {
                dep["name"]: {"id": dep["name"], "name": dep["name"]}
                for dep in all_deployments
            }

            # Convert the dictionary of unique models to a list
            model_list = list(unique_models_for_ui.values())

            # Sort the list alphabetically by the 'name' key
            sorted_model_list = sorted(model_list, key=lambda x: x["name"])

            # Return the sorted list of unique models formatted for the UI.
            return sorted_model_list
        except Exception as e:
            return [{"id": "error", "name": f"Connection Error: {e}"}]

    async def pipe(self, body: dict) -> Generator[str, None, None]:
        """
        Handles the inference request by constructing the correct API call
        based on the model family.
        """
        try:
            token = self._get_token()
            # The 'model' field from the body will now be the model name (e.g., 'sap.gpt-4o')
            raw_model_id = body.get("model", "")
            # Remove the 'sap.' prefix
            model_id = raw_model_id.split(".", 1)[-1]
            # Look up the real deployment ID using the correct model name
            deployment_id = self._get_deployment_id_for_model(model_id)
            messages = body.get("messages", [])

            # Tool context injection for consistent tool usage
            tools = body.get("tools", [])
            if tools:
                tool_names = [tool.get("name", "unknown") for tool in tools]
                tool_context = f"""You have access to the following tools: {', '.join(tool_names)}.

                IMPORTANT: When users request actions that can be performed by available tools, you MUST use the appropriate tool to perform the actual operation. Do not simulate, describe, or provide theoretical responses when a tool can do the real work.

                Guidelines:
                - Always prefer using tools over providing explanatory responses
                - Use tools to perform actual operations, not just to gather information for responses
                - If a user asks for something a tool can do, use the tool immediately
                - Be proactive in identifying when tools should be used

                Available tools: {', '.join(tool_names)}"""
                
                # Insert tool context as first system message
                messages.insert(0, {"role": "system", "content": tool_context})
                print(f"Added tool context for {len(tools)} tools: {tool_names}")

            # --- Model-specific URL and Payload Construction ---

            base_url = self.valves.AI_CORE_BASE_URL.rstrip("/")
            headers = {
                "Authorization": f"Bearer {token}",
                "AI-Resource-Group": self.valves.AI_CORE_RESOURCE_GROUP,
                "Content-Type": "application/json",
            }

            url, payload = self._build_request_params(
                base_url, deployment_id, model_id, messages, body
            )

            # --- Streaming API Call ---

            with requests.post(
                url, headers=headers, json=payload, stream=True
            ) as response:
                response.raise_for_status()
                # Yield chunks based on the model's response format
                for chunk in self._stream_response_parser(response, model_id):
                    yield chunk

        except Exception as e:
            # Yield a single error message chunk
            error_message = f"Error during model inference: {e}"
            print(f"❌ {error_message}")
            yield error_message

    # --- Request Building and Stream Parsing ---

    def _build_request_params(
        self,
        base_url: str,
        deployment_id: str,
        model_id: str,
        messages: List[Dict],
        body: Dict,
    ) -> (str, Dict):
        """Constructs the API URL and payload based on the model ID."""

        # Convert messages to OpenAI format for simplicity, as most models use a variation of it.
        openai_messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in messages
            if msg.get("role") != "system"
        ]
        system_prompt = next(
            (msg["content"] for msg in messages if msg.get("role") == "system"), None
        )

        # Classify models
        is_openai_family = any(
            prefix in model_id for prefix in ["gpt-", "o1", "o3", "o4"]
        )
        is_anthropic_family = any(
            prefix in model_id for prefix in ["anthropic--", "claude"]
        )
        is_gemini_family = "gemini" in model_id

        # OpenAI-compatible models (GPT, etc.)
        if is_openai_family:
            url = f"{base_url}/v2/inference/deployments/{deployment_id}/chat/completions?api-version=2024-12-01-preview"
            # Base payload for all OpenAI-family models
            payload = {
                "messages": (
                    [{"role": "system", "content": system_prompt}]
                    if system_prompt
                    else []
                )
                + openai_messages,
                "stream": True,
            }
            # Specific models require temperature and max_tokens to be omitted.
            if model_id in ["o1", "o3-mini", "o3", "o4-mini"]:
                # These parameters are intentionally omitted for these models.
                pass
            else:
                # Add standard parameters for other models like gpt-4o
                payload["max_tokens"] = body.get("max_tokens", self.valves.MAX_TOKENS)
                payload["temperature"] = body.get(
                    "temperature", self.valves.TEMPERATURE
                )

            return url, payload

        # Anthropic models (Claude)
        if is_anthropic_family:
            # **FIX:** Reverted to the logic that correctly identifies which models use the newer /converse-stream endpoint.
            if any(ver in model_id for ver in ["3.5", "3.7", "4"]):
                url = f"{base_url}/v2/inference/deployments/{deployment_id}/converse-stream"
                payload = {
                    "system": [{"text": system_prompt}] if system_prompt else [],
                    "messages": [
                        {"role": msg["role"], "content": [{"text": msg["content"]}]}
                        for msg in openai_messages
                    ],
                    "inference_config": {
                        "max_tokens": body.get("max_tokens", self.valves.MAX_TOKENS),
                        "temperature": body.get("temperature", self.valves.TEMPERATURE),
                    },
                }
            else:  # All other Anthropic models (including 3.0, Haiku) use the older /invoke-with-response-stream
                url = f"{base_url}/v2/inference/deployments/{deployment_id}/invoke-with-response-stream"
                payload = {
                    "system": system_prompt,
                    "messages": openai_messages,  # This format is correct for this endpoint
                    "max_tokens": body.get("max_tokens", self.valves.MAX_TOKENS),
                    "anthropic_version": "bedrock-2023-05-31",
                }
            return url, payload

        # Gemini models
        if is_gemini_family:
            url = f"{base_url}/v2/inference/deployments/{deployment_id}/models/{model_id}:streamGenerateContent"
            payload = {
                "contents": [
                    {
                        "role": "user" if msg["role"] != "assistant" else "model",
                        "parts": [{"text": msg["content"]}],
                    }
                    for msg in openai_messages
                ],
                "system_instruction": (
                    {"parts": [{"text": system_prompt}]} if system_prompt else None
                ),
                "generation_config": {
                    "max_output_tokens": body.get("max_tokens", self.valves.MAX_TOKENS),
                    "temperature": body.get("temperature", self.valves.TEMPERATURE),
                },
            }
            return url, payload

        # Default to OpenAI format if no specific family is matched
        print(
            f"Warning: Model '{model_id}' not specifically classified. Defaulting to OpenAI-compatible API format."
        )
        url = f"{base_url}/v2/inference/deployments/{deployment_id}/chat/completions?api-version=2024-12-01-preview"
        payload = {
            "messages": (
                [{"role": "system", "content": system_prompt}] if system_prompt else []
            )
            + openai_messages,
            "stream": True,
            "max_tokens": body.get("max_tokens", self.valves.MAX_TOKENS),
            "temperature": body.get("temperature", self.valves.TEMPERATURE),
        }
        return url, payload

    def _stream_response_parser(
        self, response: requests.Response, model_id: str
    ) -> Generator[str, None, None]:
        """
        Parses the SSE stream from the API and yields content.
        This version uses iter_lines for robust, line-by-line processing.
        """
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode("utf-8")
                if decoded_line.startswith("data: "):
                    json_str = decoded_line[6:]
                    if json_str.strip() == "[DONE]":
                        continue

                    is_anthropic_family = any(
                        prefix in model_id for prefix in ["anthropic--", "claude"]
                    )

                    try:
                        # Handle Anthropic's non-standard JSON response.
                        if is_anthropic_family:
                            # ast.literal_eval safely parses Python literals (like dicts with single quotes)
                            data = ast.literal_eval(json_str)
                        else:
                            # Other models use standard JSON
                            data = json.loads(json_str)

                        # --- Model-specific parsers ---
                        is_openai_family = any(
                            prefix in model_id for prefix in ["gpt-", "o1", "o3", "o4"]
                        )
                        is_gemini_family = "gemini" in model_id

                        # OpenAI Parser
                        if is_openai_family or not (
                            is_anthropic_family or is_gemini_family
                        ):
                            choices = data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                if "content" in delta and delta["content"] is not None:
                                    yield delta["content"]

                        # Anthropic Parser
                        elif is_anthropic_family:
                            # For older /invoke-with-response-stream API (Claude 3, 3.5, Haiku)
                            if data.get("type") in [
                                "content_block_start",
                                "content_block_delta",
                            ]:
                                # The actual content can be in 'content_block' or 'delta' key
                                content_block = data.get(
                                    "content_block", data.get("delta", {})
                                )
                                if (
                                    content_block.get("type") == "text_delta"
                                    or content_block.get("type") == "text"
                                ):
                                    yield content_block.get("text", "")
                            # For newer /converse-stream API (Claude 3.7, 4)
                            elif data.get("contentBlockDelta"):
                                delta = data["contentBlockDelta"].get("delta", {})
                                if "text" in delta:
                                    yield delta.get("text", "")

                        # Gemini Parser
                        elif is_gemini_family:
                            candidates = data.get("candidates", [])
                            if candidates and "content" in candidates[0]:
                                parts = candidates[0]["content"].get("parts", [])
                                if parts and "text" in parts[0]:
                                    yield parts[0]["text"]

                    except (json.JSONDecodeError, ValueError, SyntaxError) as e:
                        # Catch parsing errors from both json and ast
                        print(
                            f"Warning: Could not parse data from line: {decoded_line}. Error: {e}"
                        )
                        continue
