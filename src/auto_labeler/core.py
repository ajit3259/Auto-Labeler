import pandas as pd
from typing import List, Optional, Any
import pathlib
import yaml
from .llm import LLMAdapter

class AutoLabeler:
    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        """
        Initialize the AutoLabeler with a specific LLM model.
        """
        self.llm = LLMAdapter(model_name=model_name, api_key=api_key)
        self.prompts_dir = pathlib.Path(__file__).parent / "prompts"

    def get_usage(self) -> dict:
        """
        Returns a summary of token usage for the current session.
        """
        return self.llm.tracker.get_summary()

    def _load_prompt(self, prompt_name: str) -> str:
        with open(self.prompts_dir / f"{prompt_name}.yaml", "r") as f:
            data = yaml.safe_load(f)
            return data["template"]

    def suggest_labels(
        self, 
        df: pd.DataFrame, 
        context: str, 
        n_labels: int = 5,
        column: Optional[str] = None,
        strategy: Optional[Any] = None
    ) -> List[str]:
        """
        Suggests a list of labels based on a sample of the data and the provided context.
        If 'column' is not provided, it tries to use the first string column or all columns.
        """
        if column:
             # Logic for specific column focus can be handled in strategy in future
             pass

        # Default to Simple Strategy if none provided
        if not strategy:
            from .strategies import SimpleDiscoveryStrategy
            strategy = SimpleDiscoveryStrategy(self.llm)

        return strategy.suggest_labels(
            df=df,
            context=context,
            prompts_dir=self.prompts_dir,
            n_labels=n_labels
        )

    def label_dataset(
        self, 
        df: pd.DataFrame, 
        labels: List[str], 
        context: str, 
        target_column: str = "text",
        multi_label: bool = False,
        strategy: Optional[Any] = None,# In future type hint this better
        examples: Optional[List[dict]] = None
    ) -> pd.DataFrame:
        """
        Labels the dataset using the provided labels.
        """
        # Default to Simple Strategy if none provided
        if not strategy:
            from .strategies import SimpleLabelingStrategy
            strategy = SimpleLabelingStrategy(self.llm)
            
        return strategy.label(
            df=df, 
            labels=labels, 
            context=context, 
            prompts_dir=self.prompts_dir,
            target_column=target_column, 
            multi_label=multi_label,
            examples=examples
        )
