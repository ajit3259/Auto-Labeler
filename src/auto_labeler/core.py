import pandas as pd
from typing import List, Optional
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

    def _load_prompt(self, prompt_name: str) -> str:
        with open(self.prompts_dir / f"{prompt_name}.yaml", "r") as f:
            data = yaml.safe_load(f)
            return data["template"]

    def suggest_labels(
        self, 
        df: pd.DataFrame, 
        context: str, 
        n_labels: int = 5,
        column: Optional[str] = None
    ) -> List[str]:
        """
        Suggests a list of labels based on a sample of the data and the provided context.
        If 'column' is not provided, it tries to use the first string column or all columns.
        """
        sample = df.head(10).to_dict(orient="records")
        if column:
             # If specific column focused, just show that, but context from whole row is often good.
             # For now let's dump the whole record to give maximum context.
             pass

        prompt_template = self._load_prompt("discovery")
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
        
        response = self.llm.generate_structured(prompt, response_schema=schema)
        return response.get("labels", [])

    def label_dataset(
        self, 
        df: pd.DataFrame, 
        labels: List[str], 
        context: str, 
        target_column: str = "text",
        multi_label: bool = False
    ) -> pd.DataFrame:
        """
        Labels the dataset using the provided labels.
        
        Args:
            df: Input DataFrame
            labels: List of valid labels to choose from
            context: Context about the dataset
            target_column: The column to analyze for labeling (currently required for simplicity)
            multi_label: If True, allows multiple labels per record.
        
        Returns:
            DataFrame with a new 'predicted_label' column.
        """
        # Pragmatic: Loop through for MVP. Batching is a V2 feature.
        # We'll use a copy to avoid SettingWithCopy warnings
        result_df = df.copy()
        
        label_results = []
        prompt_template = self._load_prompt("assignment")
        
        # Prepare instruction strings once
        multi_label_instruction = 'Select strictly one label.' if not multi_label else 'Select one or more labels.'
        output_format_instruction = 'a string' if not multi_label else 'a list of strings'

        # We can optimize by batching later, but for now row-by-row is safest for error handling
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
                # We don't enforce strict schema validation here for every row to save latency/tokens 
                # effectively, but we do ask for structured output.
                # A simple generate_structured call is robust enough.
                response = self.llm.generate_structured(prompt, response_schema={})
                assigned_label = response.get("label")
                
                # Basic validation
                if not multi_label and isinstance(assigned_label, list):
                    assigned_label = assigned_label[0] if assigned_label else None
                
                label_results.append(assigned_label)
            except Exception as e:
                # Fail gracefully
                # print(f"Error labeling row {index}: {e}")
                label_results.append(None)
        
        result_df['predicted_label'] = label_results
        return result_df
