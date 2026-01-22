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

class IterativeDiscoveryStrategy:
    """
    Advanced iterative discovery strategy supporting multiple modes:
    1. 'refine' (Default): Seed -> Sweep (Find 'Other') -> Refine. Good for finding edge cases.
    2. 'evolve': Sequential batches. Batch 1 -> Labels A. Batch 2 + Labels A -> Labels B. Good for concept drift.
    3. 'aggregate': Independent batches -> Merge. Good for parallelization.
    """
    def __init__(
        self, 
        llm: LLMAdapter, 
        mode: str = "refine",
        seed_sample_size: int = 10, 
        batch_size: int = 50, 
        other_threshold: int = 5
    ):
        self.llm = llm
        self.mode = mode
        self.seed_sample_size = seed_sample_size
        self.batch_size = batch_size
        self.other_threshold = other_threshold

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
        if self.mode == "evolve":
            return self._run_evolve(df, context, prompts_dir, n_labels)
        elif self.mode == "aggregate":
            return self._run_aggregate(df, context, prompts_dir, n_labels)
        else:
            return self._run_refine(df, context, prompts_dir, n_labels)

    def _run_refine(self, df: pd.DataFrame, context: str, prompts_dir: pathlib.Path, n_labels: int) -> List[str]:
        # --- Phase 1: Seed ---
        simple_strategy = SimpleDiscoveryStrategy(self.llm, sample_size=self.seed_sample_size, shuffle=True)
        seed_labels = simple_strategy.suggest_labels(df, context, prompts_dir, n_labels)
        
        if len(df) <= self.seed_sample_size:
            return seed_labels

        # --- Phase 2: Sweep (Validation) ---
        validation_n = min(len(df), self.batch_size)
        validation_df = df.sample(validation_n)
        validation_records = validation_df.to_dict(orient="records")

        classify_prompt = self._load_prompt(prompts_dir, "discovery_classify")
        prompt = classify_prompt.format(
            context=context,
            labels=seed_labels,
            items=validation_records
        )
        
        other_items = []
        try:
            schema = {
                "type": "object",
                "properties": {
                    "other_items": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["other_items"]
            }
            response = self.llm.generate_structured(prompt, response_schema=schema)
            other_items = response.get("other_items", [])
        except Exception:
            pass

        # --- Phase 3: Refine ---
        final_labels = list(seed_labels)
        
        if len(other_items) >= self.other_threshold:
            other_df = pd.DataFrame({"text": other_items})
            refinement_strategy = SimpleDiscoveryStrategy(self.llm, sample_size=len(other_df))
            new_labels = refinement_strategy.suggest_labels(other_df, context, prompts_dir, n_labels=3)
            
            for label in new_labels:
                if label not in final_labels:
                    final_labels.append(label)
        
        return final_labels[:n_labels] if len(final_labels) > n_labels else final_labels

    def _run_evolve(self, df: pd.DataFrame, context: str, prompts_dir: pathlib.Path, n_labels: int) -> List[str]:
        # Split dataframe into chunks of batch_size
        chunks = [df[i:i + self.batch_size] for i in range(0, len(df), self.batch_size)]
        current_labels = []
        
        for chunk in chunks:
            # We treat the 'Existing Labels' as part of the context for the simple discovery prompt
            # But SimpleDiscovery doesn't take 'existing_labels' param. 
            # We simulate evolution by appending existing labels to context or using a specific prompt.
            # Simpler approach: Just run SimpleDiscovery on the chunk, then Merge with current.
            # Ideally, we'd tell the LLM "Here are current labels, improve them", but let's stick to Merge for stability.
            
            chunk_strategy = SimpleDiscoveryStrategy(self.llm, sample_size=len(chunk))
            # We append current labels to context to guide the LLM (Soft evolution)
            evolved_context = f"{context}\n\nExisting Known Labels: {current_labels}"
            new_labels = chunk_strategy.suggest_labels(chunk, evolved_context, prompts_dir, n_labels)
            
            # Simple union update
            for label in new_labels:
                if label not in current_labels:
                    current_labels.append(label)
        
        return current_labels[:n_labels]

    def _run_aggregate(self, df: pd.DataFrame, context: str, prompts_dir: pathlib.Path, n_labels: int) -> List[str]:
        # Split into chunks, get labels for each independently
        chunks = [df[i:i + self.batch_size] for i in range(0, len(df), self.batch_size)]
        all_label_lists = []
        
        for chunk in chunks:
            chunk_strategy = SimpleDiscoveryStrategy(self.llm, sample_size=len(chunk))
            labels = chunk_strategy.suggest_labels(chunk, context, prompts_dir, n_labels)
            all_label_lists.append(labels)
            
        # Merge Step
        merge_prompt_template = self._load_prompt(prompts_dir, "discovery_merge")
        prompt = merge_prompt_template.format(
            context=context,
            label_lists=all_label_lists
        )
        
        try:
            schema = {
                "type": "object",
                "properties": {
                    "labels": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["labels"]
            }
            response = self.llm.generate_structured(prompt, response_schema=schema)
            final_labels = response.get("labels", [])
            return final_labels
        except Exception:
            # Fallback: Flatten and distinct
            flat_labels = list(set([lbl for sublist in all_label_lists for lbl in sublist]))
            return flat_labels[:n_labels]
