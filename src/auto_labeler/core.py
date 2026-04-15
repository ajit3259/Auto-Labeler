import pandas as pd
from typing import List, Optional, Any, Dict
import pathlib
import yaml
from .llm import LLMAdapter
from .schemas import AutoLabelerConfig, LabelingConfig, DiscoveryConfig
from .logger import logger, setup_logger

class AutoLabeler:
    """
    Main entry point for the Auto-Labeler library.
    Orchestrates discovery and labeling tasks using various strategies.
    """
    def __init__(
        self, 
        model_name: str = "gemini/gemini-2.5-flash", 
        api_key: Optional[str] = None,
        use_cache: bool = True,
        cache_dir: str = ".auto_labeler_cache",
        log_level: str = "INFO"
    ):
        """
        Initialize the AutoLabeler with model configuration and preferences.
        """
        # Set log level
        setup_logger(level=log_level)
        
        self.config = AutoLabelerConfig(
            model_name=model_name,
            api_key=api_key,
            use_cache=use_cache,
            cache_dir=cache_dir,
            log_level=log_level
        )
        
        self.llm = LLMAdapter(
            model_name=self.config.model_name, 
            api_key=self.config.api_key,
            use_cache=self.config.use_cache,
            cache_dir=self.config.cache_dir
        )
        self.prompts_dir = pathlib.Path(__file__).parent / "prompts"

    def get_usage(self) -> dict:
        """
        Returns a summary of token usage and estimated cost for the current session.
        """
        return self.llm.tracker.get_summary()

    def _validate_df(self, df: pd.DataFrame, column: Optional[str] = None):
        """Internal validation for dataframes."""
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
        if column and column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame.")

    def suggest_labels(
        self, 
        df: pd.DataFrame, 
        context: str, 
        n_labels: int = 5,
        column: Optional[str] = None,
        strategy: Optional[Any] = None
    ) -> List[str]:
        """
        Suggests a list of labels based on the dataset content and context.
        """
        # Validate inputs via Pydantic
        config = DiscoveryConfig(context=context, n_labels=n_labels)
        self._validate_df(df, column)
        
        logger.info(f"Starting label discovery (asking for {config.n_labels} labels)...")

        # Default to Simple Strategy if none provided
        if not strategy:
            from .strategies import SimpleDiscoveryStrategy
            strategy = SimpleDiscoveryStrategy(self.llm)

        return strategy.suggest_labels(
            df=df,
            context=config.context,
            prompts_dir=self.prompts_dir,
            n_labels=config.n_labels
        )

    def label_dataset(
        self, 
        df: pd.DataFrame, 
        labels: List[str], 
        context: str, 
        target_column: str = "text",
        multi_label: bool = False,
        batch_size: int = 1,
        strategy: Optional[Any] = None,
        examples: Optional[List[dict]] = None
    ) -> pd.DataFrame:
        """
        Labels the dataset using the provided labels and context.
        """
        # Validate inputs via Pydantic
        config = LabelingConfig(
            context=context, 
            labels=labels, 
            target_column=target_column, 
            multi_label=multi_label,
            batch_size=batch_size
        )
        self._validate_df(df, config.target_column)

        logger.info(f"Starting labeling task on {len(df)} records (Batch Size: {config.batch_size})...")

        # Default to Simple Strategy if none provided
        if not strategy:
            from .strategies import SimpleLabelingStrategy
            strategy = SimpleLabelingStrategy(self.llm, batch_size=config.batch_size)
        elif hasattr(strategy, 'batch_size'):
            # Override strategy batch size if provided in call
            strategy.batch_size = config.batch_size
            
        return strategy.label(
            df=df, 
            labels=config.labels, 
            context=config.context, 
            prompts_dir=self.prompts_dir,
            target_column=config.target_column, 
            multi_label=config.multi_label,
            examples=examples
        )
