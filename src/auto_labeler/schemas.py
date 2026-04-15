from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Union, Any
import pathlib

class LabelingConfig(BaseModel):
    """Configuration for labeling operations."""
    context: str = Field(..., description="Description of the dataset context.")
    labels: List[str] = Field(..., min_items=1, description="List of allowed labels.")
    target_column: str = Field("text", description="Column containing text to label.")
    multi_label: bool = Field(False, description="Whether to allow multiple labels per record.")
    batch_size: int = Field(1, ge=1, description="Number of records per LLM call.")

class DiscoveryConfig(BaseModel):
    """Configuration for label discovery operations."""
    context: str = Field(..., description="Description of the dataset context.")
    n_labels: int = Field(5, ge=1, description="Number of labels to suggest.")
    sample_size: int = Field(50, ge=1, description="Number of records to sample for discovery.")

class AutoLabelerConfig(BaseModel):
    """Global configuration for the AutoLabeler instance."""
    model_name: str = Field("gemini/gemini-flash-latest")
    api_key: Optional[str] = None
    use_cache: bool = True
    cache_dir: str = ".auto_labeler_cache"
    log_level: str = "INFO"
