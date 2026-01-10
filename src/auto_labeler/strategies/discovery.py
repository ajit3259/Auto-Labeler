from typing import List, Protocol
import pandas as pd
from ..llm import LLMAdapter
import pathlib
import yaml

class DiscoveryStrategy(Protocol):
    def suggest_labels(
        self, 
        df: pd.DataFrame, 
        context: str, 
        prompts_dir: pathlib.Path, 
        n_labels: int = 5
    ) -> List[str]:
        ...

class SimpleDiscoveryStrategy:
    def __init__(self, llm: LLMAdapter):
        self.llm = llm

    def _load_prompt(self, prompts_dir: pathlib.Path, prompt_name: str) -> str:
        with open(prompts_dir / f"{prompt_name}.yaml", "r") as f:
            data = yaml.safe_load(f)
            return data["template"]

    def suggest_labels(
        self, 
        df: pd.DataFrame, 
        context: str, 
        prompts_dir: pathlib.Path, 
        n_labels: int = 5
    ) -> List[str]:
        # Simple head(10) approach
        sample = df.head(10).to_dict(orient="records")
        
        prompt_template = self._load_prompt(prompts_dir, "discovery")
        prompt = prompt_template.format(
            context=context,
            sample=sample,
            n_labels=n_labels
        )
        
        schema = {
            "type": "object",
            "properties": {
                "labels": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["labels"]
        }
        
        try:
            response = self.llm.generate_structured(prompt, response_schema=schema)
            return response.get("labels", [])
        except Exception:
            return []

class ParallelDiscoveryStrategy:
    def __init__(self, llm: LLMAdapter, num_samples: int = 3, sample_size: int = 10):
        self.llm = llm
        self.num_samples = num_samples
        self.sample_size = sample_size

    def _load_prompt(self, prompts_dir: pathlib.Path, prompt_name: str) -> str:
        with open(prompts_dir / f"{prompt_name}.yaml", "r") as f:
            data = yaml.safe_load(f)
            return data["template"]

    def suggest_labels(
        self, 
        df: pd.DataFrame, 
        context: str, 
        prompts_dir: pathlib.Path, 
        n_labels: int = 5
    ) -> List[str]:
        all_labels = set()
        prompt_template = self._load_prompt(prompts_dir, "discovery")
        
        for _ in range(self.num_samples):
            # Sample with replacement allowed if df is small, otherwise distinct
            if len(df) > self.sample_size:
                sample = df.sample(self.sample_size).to_dict(orient="records")
            else:
                sample = df.to_dict(orient="records")
            
            prompt = prompt_template.format(
                context=context,
                sample=sample,
                n_labels=n_labels
            )
            
            schema = {
                "type": "object",
                "properties": {
                    "labels": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["labels"]
            }
            
            try:
                response = self.llm.generate_structured(prompt, response_schema=schema)
                labels = response.get("labels", [])
                all_labels.update(labels)
            except Exception:
                continue
        
        return list(all_labels)
