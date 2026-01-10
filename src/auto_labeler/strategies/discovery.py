from typing import List, Protocol
import pandas as pd
from ..llm import LLMAdapter
import pathlib
import yaml

class DiscoveryStrategy(Protocol):
    """
    Protocol for label discovery strategies.
    Responsible for looking at data samples and suggesting potential labels.
    """
    def suggest_labels(
        self, 
        df: pd.DataFrame, 
        context: str, 
        prompts_dir: pathlib.Path, 
        n_labels: int = 5
    ) -> List[str]:
        """
        Suggest potential labels for the dataset.
        
        Args:
            df: Input dataframe.
            context: Description of the dataset context.
            prompts_dir: Directory containing YAML prompts.
            n_labels: Number of labels to suggest.
            
        Returns:
            List of suggested label strings.
        """
        ...

class SimpleDiscoveryStrategy:
    """
    Standard discovery strategy that analyzes a sample of the dataframe.
    Defaults to the first 10 rows (head) but can shuffle.
    """
    def __init__(self, llm: LLMAdapter, sample_size: int = 10, shuffle: bool = False):
        self.llm = llm
        self.sample_size = sample_size
        self.shuffle = shuffle

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
        # Sample selection logic
        if self.shuffle:
            # Safe sample (handle if df < sample_size)
            n = min(len(df), self.sample_size)
            sample_df = df.sample(n)
        else:
            sample_df = df.head(self.sample_size)
            
        sample = sample_df.to_dict(orient="records")
        
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
    """
    Advanced discovery strategy that samples the dataset multiple times in parallel 
    to increase surface area coverage.
    """
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
        
        return list(all_labels)[:n_labels] if len(all_labels) > n_labels else list(all_labels)
