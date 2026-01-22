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
    "Clustering as Classification" strategy.
    1. Seed: Discover labels from a small sample.
    2. Sweep: Classify a larger sample against these labels.
    3. Refine: Collect 'Other' items and discover new labels from them.
    4. Merge: Combine Seed and Refined labels.
    """
    def __init__(self, llm: LLMAdapter, seed_sample_size: int = 10, validation_sample_size: int = 50, other_threshold: int = 5):
        self.llm = llm
        self.seed_sample_size = seed_sample_size
        self.validation_sample_size = validation_sample_size
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
        # --- Phase 1: Seed ---
        # Reuse SimpleDiscovery logic for the seed
        simple_strategy = SimpleDiscoveryStrategy(self.llm, sample_size=self.seed_sample_size, shuffle=True)
        seed_labels = simple_strategy.suggest_labels(df, context, prompts_dir, n_labels)
        
        # If dataset is small, just return seed labels
        if len(df) <= self.seed_sample_size:
            return seed_labels

        # --- Phase 2: Sweep (Validation) ---
        # Sample a larger set (excluding the seed ideally, but random sampling is fine for approx)
        validation_n = min(len(df), self.validation_sample_size)
        validation_df = df.sample(validation_n)
        validation_records = validation_df.to_dict(orient="records")

        # We need a "Classifier" prompt here. 
        # For simplicity in this iteration, we will ask the LLM to "Bin" the items.
        # "For each item, assign it to one of {seed_labels} or 'Other'."
        
        # Construct a mini-prompt for classification
        classify_prompt_template = """
        You are a data labeler.
        Context: {context}
        Existing Labels: {labels}
        
        Task: Classify the following VALIDATION ITEMS into the Existing Labels.
        If an item clearly does not fit any existing label, classify it as "Other".
        
        Validation Items:
        {items}
        
        Return a JSON object: {{"other_items": ["text of item 1", "text of item 2"]}}
        Only return items classified as "Other".
        """
        
        # We might need to batch this if validation_sample_size is large, but assuming 50 fits in context.
        # Let's just process it.
        
        prompt = classify_prompt_template.format(
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
            # If classification fails, assume no new info found
            pass

        # --- Phase 3: Refine ---
        final_labels = list(seed_labels)
        
        if len(other_items) >= self.other_threshold:
            # We found enough "Other" items to warrant a new discovery pass
            # Treat 'other_items' as a new dataframe
            other_df = pd.DataFrame({"text": other_items}) # Assuming 'text' column, but list of strings is fine
            
            # Run simple discovery on these specific items
            # We want to find *new* labels, so passing n_labels (e.g. 2-3)
            refinement_strategy = SimpleDiscoveryStrategy(self.llm, sample_size=len(other_df))
            new_labels = refinement_strategy.suggest_labels(other_df, context, prompts_dir, n_labels=3)
            
            # --- Phase 4: Merge ---
            # Add new labels if they are not duplicates
            for label in new_labels:
                if label not in final_labels:
                    final_labels.append(label)
        
        return final_labels[:n_labels] if len(final_labels) > n_labels else final_labels
