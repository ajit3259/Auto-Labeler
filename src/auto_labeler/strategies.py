from typing import List, Protocol, Any, Dict, Optional, Union
import pandas as pd
from .llm import LLMAdapter
import pathlib
import yaml

class LabelingStrategy(Protocol):
    def label(
        self, 
        df: pd.DataFrame, 
        labels: List[str], 
        context: str, 
        prompts_dir: pathlib.Path,
        target_column: str = "text",
        multi_label: bool = False
    ) -> pd.DataFrame:
        ...

class SimpleLabelingStrategy:
    def __init__(self, llm: LLMAdapter):
        self.llm = llm

    def _load_prompt(self, prompts_dir: pathlib.Path, prompt_name: str) -> str:
        # Duplicated logic for now, can perform cleanup later
        with open(prompts_dir / f"{prompt_name}.yaml", "r") as f:
            data = yaml.safe_load(f)
            return data["template"]

    def label(
        self, 
        df: pd.DataFrame, 
        labels: List[str], 
        context: str, 
        prompts_dir: pathlib.Path,
        target_column: str = "text",
        multi_label: bool = False
    ) -> pd.DataFrame:
        result_df = df.copy()
        label_results = []
        prompt_template = self._load_prompt(prompts_dir, "assignment")
        
        multi_label_instruction = 'Select strictly one label.' if not multi_label else 'Select one or more labels.'
        output_format_instruction = 'a string' if not multi_label else 'a list of strings'

        for index, row in result_df.iterrows():
            record_content = row[target_column] if target_column in row else str(row.to_dict())
            
            prompt = prompt_template.format(
                context=context,
                labels=labels,
                record_content=record_content,
                multi_label_instruction=multi_label_instruction,
                output_format_instruction=output_format_instruction
            )
            
            try:
                response = self.llm.generate_structured(prompt, response_schema={})
                assigned_label = response.get("label")
                
                if not multi_label and isinstance(assigned_label, list):
                    assigned_label = assigned_label[0] if assigned_label else None
                
                label_results.append(assigned_label)
            except Exception as e:
                label_results.append(None)
        
        result_df['predicted_label'] = label_results
        return result_df

class ConsensusLabelingStrategy:
    def __init__(self, models: List[str], adjudicator_model: str, api_key: Optional[str] = None):
        """
        Args:
            models: List of model names to use as judges (e.g. ['gpt-3.5-turbo', 'gemini-1.5-flash', ...])
            adjudicator_model: Strong model to decide in case of disagreement.
        """
        # Create an adapter for each model
        self.adapters = [LLMAdapter(model_name=m, api_key=api_key) for m in models]
        self.adjudicator = LLMAdapter(model_name=adjudicator_model, api_key=api_key)

    def _load_prompt(self, prompts_dir: pathlib.Path, prompt_name: str) -> str:
        with open(prompts_dir / f"{prompt_name}.yaml", "r") as f:
            data = yaml.safe_load(f)
            return data["template"]

    def label(
        self, 
        df: pd.DataFrame, 
        labels: List[str], 
        context: str, 
        prompts_dir: pathlib.Path,
        target_column: str = "text",
        multi_label: bool = False
    ) -> pd.DataFrame:
        result_df = df.copy()
        label_results = []
        confidence_results = []
        
        assignment_template = self._load_prompt(prompts_dir, "assignment")
        adjudicator_template = self._load_prompt(prompts_dir, "adjudicator")
        
        multi_label_instruction = 'Select strictly one label.' if not multi_label else 'Select one or more labels.'
        output_format_instruction = 'a string' if not multi_label else 'a list of strings'

        for index, row in result_df.iterrows():
            record_content = row[target_column] if target_column in row else str(row.to_dict())
            
            # 1. BroadCast Query
            prompt = assignment_template.format(
                context=context,
                labels=labels,
                record_content=record_content,
                multi_label_instruction=multi_label_instruction,
                output_format_instruction=output_format_instruction
            )
            
            votes = []
            for adapter in self.adapters:
                try:
                    # In a real impl, these should be parallelized (asyncio or ThreadPool)
                    # For MVP simplistic sequential is fine for proof of concept
                    response = adapter.generate_structured(prompt, response_schema={})
                    val = response.get("label")
                    # Normalize single items
                    if not multi_label and isinstance(val, list):
                        val = val[0] if val else None
                    votes.append(val)
                except Exception as e:
                    print(f"Error in consensus vote: {e}")
                    import traceback
                    traceback.print_exc()
                    votes.append(None)
            
            # 2. Check Consensus
            # Filter None
            valid_votes = [v for v in votes if v is not None]
            
            if not valid_votes:
                label_results.append(None)
                confidence_results.append("Failed")
                continue

            # Check if all elements are equal (first element == all others)
            # Handling lists (multi-label) requires careful comparison, assuming strictly sorted or identical order
            # For MVP, string comparison of representation is an easy hack, or set comparison
            
            is_unanimous = all(v == valid_votes[0] for v in valid_votes)
            
            if is_unanimous:
                label_results.append(valid_votes[0])
                confidence_results.append("High (Unanimous)")
            else:
                # 3. Adjudicate
                vote_str = "\n".join([f"Model {i+1}: {v}" for i, v in enumerate(votes)])
                adj_prompt = adjudicator_template.format(
                    context=context,
                    labels=labels,
                    record_content=record_content,
                    votes=vote_str
                )
                try:
                    adj_response = self.adjudicator.generate_structured(adj_prompt, response_schema={})
                    final_label = adj_response.get("label")
                    label_results.append(final_label)
                    confidence_results.append("Medium (Adjudicated)")
                except Exception:
                    # Fallback to majority vote? Or just first valid?
                    # Fallback to first valid vote
                    label_results.append(valid_votes[0])
                    confidence_results.append("Low (Fallback)")

        result_df['predicted_label'] = label_results
        result_df['confidence_level'] = confidence_results
        return result_df

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
                # Parallelize this in V3
                response = self.llm.generate_structured(prompt, response_schema=schema)
                labels = response.get("labels", [])
                all_labels.update(labels)
            except Exception:
                continue
        
        return list(all_labels)
