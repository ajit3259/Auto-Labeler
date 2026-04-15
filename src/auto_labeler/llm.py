import json
from typing import Any, Dict, List, Optional, Type, Union
import litellm
import asyncio
from pydantic import BaseModel

class UsageTracker:
    """
    Stays updated with the cumulative token usage for a session.
    """
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def update(self, usage: Any):
        if hasattr(usage, 'prompt_tokens'):
            self.prompt_tokens += usage.prompt_tokens
        if hasattr(usage, 'completion_tokens'):
            self.completion_tokens += usage.completion_tokens
        if hasattr(usage, 'total_tokens'):
            self.total_tokens += usage.total_tokens

    def get_summary(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens
        }

class LLMAdapter:
    """
    A unified wrapper around LiteLLM to handle different providers (OpenAI, Gemini, etc.)
    and standard interactions like text generation and structured outputs.
    """
    def __init__(self, model_name: str, api_key: Optional[str] = None, tracker: Optional[UsageTracker] = None):
        """
        Args:
            model_name: The LiteLLM model identifier (e.g. 'gpt-3.5-turbo').
            api_key: Optional API key override.
            tracker: Optional UsageTracker to aggregate token usage.
        """
        self.model_name = model_name
        self.api_key = api_key
        self.tracker = tracker if tracker else UsageTracker()

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generates a simple text response from the LLM.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = litellm.completion(
            model=self.model_name,
            api_key=self.api_key,
            messages=messages
        )
        self.tracker.update(response.usage)
        return response.choices[0].message.content

    async def agenerate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Asynchronous version of generate.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await litellm.acompletion(
            model=self.model_name,
            api_key=self.api_key,
            messages=messages
        )
        self.tracker.update(response.usage)
        return response.choices[0].message.content

    def generate_structured(
        self, 
        prompt: str, 
        response_schema: Dict[str, Any], 
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generates a structured JSON response based on a schema.
        Uses JSON mode where available or prompts the model to output JSON.
        """
        # Append instruction to output JSON if not present
        if "json" not in prompt.lower():
            prompt += "\n\nPlease output the result as a raw JSON object."

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Litellm supports response_format with schema for many providers
        response = litellm.completion(
            model=self.model_name,
            api_key=self.api_key,
            messages=messages,
            response_format=response_schema if response_schema else {"type": "json_object"},
            num_retries=3
        )
        self.tracker.update(response.usage)
        
        content = response.choices[0].message.content
        return self._parse_json_content(content)

    async def agenerate_structured(
        self, 
        prompt: str, 
        response_schema: Dict[str, Any], 
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Asynchronous version of generate_structured.
        """
        if "json" not in prompt.lower():
            prompt += "\n\nPlease output the result as a raw JSON object."

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = await litellm.acompletion(
            model=self.model_name,
            api_key=self.api_key,
            messages=messages,
            response_format=response_schema if response_schema else {"type": "json_object"},
            num_retries=3
        )
        self.tracker.update(response.usage)
        
        content = response.choices[0].message.content
        return self._parse_json_content(content)

    def _parse_json_content(self, content: str) -> Dict[str, Any]:
        # Clean markdown code blocks if present
        clean_content = content.strip()
        if clean_content.startswith("```"):
            # Remove ```json or just ```
            lines = clean_content.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            clean_content = "\n".join(lines).strip()
            
        try:
            return json.loads(clean_content)
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse LLM response as JSON: {clean_content}")
    
    def get_embedding(self, text: Union[str, List[str]], model: str = "text-embedding-3-small") -> Union[List[float], List[List[float]]]:
        """
        Generates embeddings for a string or list of strings.
        """
        response = litellm.embedding(
            model=model,
            input=text,
            api_key=self.api_key
        )
        # Assuming embedding also has usage
        if hasattr(response, 'usage'):
            self.tracker.update(response.usage)
            
        # Handle both single string and list inputs
        data = response.data
        if isinstance(text, str):
            return data[0]["embedding"]
        else:
            return [item["embedding"] for item in data]
