import json
import hashlib
from typing import Any, Dict, List, Optional, Type, Union
import litellm
import asyncio
from pydantic import BaseModel
from .logger import logger

try:
    from diskcache import Cache
except ImportError:
    Cache = None

class UsageTracker:
    """
    Stays updated with the cumulative token usage and estimated cost for a session.
    """
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.total_cost_usd = 0.0

    def update(self, response: Any):
        """
        Updates usage from a LiteLLM response object.
        """
        usage = getattr(response, "usage", None)
        if usage:
            self.prompt_tokens += getattr(usage, 'prompt_tokens', 0)
            self.completion_tokens += getattr(usage, 'completion_tokens', 0)
            self.total_tokens += getattr(usage, 'total_tokens', 0)
        
        # Calculate cost using litellm
        try:
            cost = litellm.completion_cost(completion_response=response)
            self.total_cost_usd += float(cost) if cost else 0.0
        except Exception:
            # Fallback if cost calculation fails
            pass

    def get_summary(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6)
        }

class LLMAdapter:
    """
    A unified wrapper around LiteLLM to handle different providers (OpenAI, Gemini, etc.)
    and standard interactions like text generation and structured outputs.
    Supports persistent disk caching of responses.
    """
    def __init__(
        self, 
        model_name: str, 
        api_key: Optional[str] = None, 
        tracker: Optional[UsageTracker] = None,
        use_cache: bool = True,
        cache_dir: str = ".auto_labeler_cache"
    ):
        """
        Args:
            model_name: The LiteLLM model identifier (e.g. 'gpt-3.5-turbo').
            api_key: Optional API key override.
            tracker: Optional UsageTracker to aggregate token usage.
            use_cache: Whether to use disk caching for responses.
            cache_dir: Directory to store the cache.
        """
        self.model_name = model_name
        self.api_key = api_key
        self.tracker = tracker if tracker else UsageTracker()
        self.use_cache = use_cache
        self.cache = None
        
        if use_cache and Cache:
            self.cache = Cache(cache_dir)
            logger.debug(f"Caching enabled. Storage: {cache_dir}")
        elif use_cache and not Cache:
            logger.warning("diskcache not installed. Caching will be disabled.")

    def _get_cache_key(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Generates a stable cache key for a request."""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system_prompt": system_prompt,
            **kwargs
        }
        encoded = json.dumps(payload, sort_keys=True).encode()
        key = hashlib.md5(encoded).hexdigest()
        logger.debug(f"Generated cache key for {self.model_name}: {key}")
        return key

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generates a simple text response from the LLM.
        """
        cache_key = self._get_cache_key(prompt, system_prompt)
        if self.cache is not None:
            cached_val = self.cache.get(cache_key)
            if cached_val is not None:
                logger.debug("Cache hit for simple generation.")
                return cached_val

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = litellm.completion(
            model=self.model_name,
            api_key=self.api_key,
            messages=messages
        )
        self.tracker.update(response)
        result = response.choices[0].message.content
        
        if self.cache is not None:
            self.cache[cache_key] = result
        return result

    async def agenerate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Asynchronous version of generate.
        """
        cache_key = self._get_cache_key(prompt, system_prompt, async_call=True)
        if self.cache is not None:
            cached_val = self.cache.get(cache_key)
            if cached_val is not None:
                return cached_val

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await litellm.acompletion(
            model=self.model_name,
            api_key=self.api_key,
            messages=messages
        )
        self.tracker.update(response)
        result = response.choices[0].message.content
        
        if self.cache is not None:
            self.cache[cache_key] = result
        return result

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
        cache_key = self._get_cache_key(prompt, system_prompt, schema=response_schema)
        if self.cache is not None:
            cached_val = self.cache.get(cache_key)
            if cached_val is not None:
                logger.debug("Cache hit for structured generation.")
                return cached_val

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
        self.tracker.update(response)
        
        content = response.choices[0].message.content
        result = self._parse_json_content(content)
        
        if self.cache is not None:
            self.cache[cache_key] = result
        return result

    async def agenerate_structured(
        self, 
        prompt: str, 
        response_schema: Dict[str, Any], 
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Asynchronous version of generate_structured.
        """
        cache_key = self._get_cache_key(prompt, system_prompt, schema=response_schema, async_call=True)
        if self.cache is not None:
            cached_val = self.cache.get(cache_key)
            if cached_val is not None:
                return cached_val

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
        self.tracker.update(response)
        
        content = response.choices[0].message.content
        result = self._parse_json_content(content)
        
        if self.cache is not None:
            self.cache[cache_key] = result
        return result

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
        cache_key = self._get_cache_key(str(text), model=model, type="split_embedding")
        if self.cache and cache_key in self.cache:
            return self.cache[cache_key]

        response = litellm.embedding(
            model=model,
            input=text,
            api_key=self.api_key
        )
        self.tracker.update(response)
            
        # Handle both single string and list inputs
        data = response.data
        if isinstance(text, str):
            result = data[0]["embedding"]
        else:
            result = [item["embedding"] for item in data]
            
        if self.cache is not None:
            self.cache[cache_key] = result
        return result
