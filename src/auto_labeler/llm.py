import json
from typing import Any, Dict, List, Optional, Type, Union
import litellm
from pydantic import BaseModel

class LLMAdapter:
    def __init__(self, model_name: str, api_key: Optional[str] = None):
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
        
        # Litellm supports response_format={"type": "json_object"} for many providers
        response = litellm.completion(
            model=self.model_name,
            api_key=self.api_key,
            messages=messages,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Fallback or simple error handling for now
            raise ValueError(f"Failed to parse LLM response as JSON: {content}")
