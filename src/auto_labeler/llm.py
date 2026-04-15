import json
from typing import Any, Dict, List, Optional, Type, Union
import litellm
from pydantic import BaseModel

class LLMAdapter:
    """
    A unified wrapper around LiteLLM to handle different providers (OpenAI, Gemini, etc.)
    and standard interactions like text generation and structured outputs.
    """
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        """
        Args:
            model_name: The LiteLLM model identifier (e.g. 'gpt-3.5-turbo').
            api_key: Optional API key override.
        """
        self.model_name = model_name
        self.api_key = api_key

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
        
        content = response.choices[0].message.content
        
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
            # Fallback or simple error handling for now
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
        # Handle both single string and list inputs
        data = response.data
        if isinstance(text, str):
            return data[0]["embedding"]
        else:
            return [item["embedding"] for item in data]
