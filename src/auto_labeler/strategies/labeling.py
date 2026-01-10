from typing import List, Protocol, Optional, Union, Dict
import pandas as pd
from ..llm import LLMAdapter
import pathlib
import yaml
from jinja2 import Template

class LabelingStrategy(Protocol):
    def label(
        self, 
        df: pd.DataFrame, 
        labels: List[str], 
        context: str, 
        prompts_dir: pathlib.Path,
        target_column: str = "text",
        multi_label: bool = False,
        examples: Optional[List[Dict[str, str]]] = None
    ) -> pd.DataFrame:
        ...

class SimpleLabelingStrategy:
    def __init__(self, llm: LLMAdapter):
        self.llm = llm

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
        multi_label: bool = False,
        examples: Optional[List[Dict[str, str]]] = None
    ) -> pd.DataFrame:
        result_df = df.copy()
        label_results = []
        prompt_template = self._load_prompt(prompts_dir, "assignment")
        
        multi_label_instruction = 'Select strictly one label.' if not multi_label else 'Select one or more labels.'
        output_format_instruction = 'a string' if not multi_label else 'a list of strings'
        
        template = Template(prompt_template)
        
        for index, row in result_df.iterrows():
            record_content = row[target_column] if target_column in row else str(row.to_dict())
            
            # Render with Jinja2
            prompt = template.render(
                context=context,
                labels=labels,
                record_content=record_content,
                multi_label_instruction=multi_label_instruction,
                output_format_instruction=output_format_instruction,
                examples=examples
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
        multi_label: bool = False,
        examples: Optional[List[Dict[str, str]]] = None
    ) -> pd.DataFrame:
        result_df = df.copy()
        label_results = []
        confidence_results = []
        
        assignment_template = self._load_prompt(prompts_dir, "assignment")
        adjudicator_template = self._load_prompt(prompts_dir, "adjudicator")
        
        multi_label_instruction = 'Select strictly one label.' if not multi_label else 'Select one or more labels.'
        output_format_instruction = 'a string' if not multi_label else 'a list of strings'
        
        from jinja2 import Template
        template = Template(assignment_template)

        for index, row in result_df.iterrows():
            record_content = row[target_column] if target_column in row else str(row.to_dict())
            
            # 1. BroadCast Query
            prompt = template.render(
                context=context,
                labels=labels,
                record_content=record_content,
                multi_label_instruction=multi_label_instruction,
                output_format_instruction=output_format_instruction,
                examples=examples
            )
            
            votes = []
            for adapter in self.adapters:
                try:
                    response = adapter.generate_structured(prompt, response_schema={})
                    val = response.get("label")
                    if not multi_label and isinstance(val, list):
                        val = val[0] if val else None
                    votes.append(val)
                except Exception as e:
                    print(f"Error in consensus vote: {e}")
                    votes.append(None)
            
            # 2. Check Consensus
            valid_votes = [v for v in votes if v is not None]
            
            if not valid_votes:
                label_results.append(None)
                confidence_results.append("Failed")
                continue

            is_unanimous = all(v == valid_votes[0] for v in valid_votes)
            
            if is_unanimous:
                label_results.append(valid_votes[0])
                confidence_results.append("High (Unanimous)")
            else:
                # 3. Adjudicate
                vote_str = "\n".join([f"Model {i+1}: {v}" for i, v in enumerate(votes)])
                # Adjudicator prompt doesn't need examples usually, but could benefit. 
                # For now keep simple format/replace for adjudicator as we didn't update that yaml.
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
                    label_results.append(valid_votes[0])
                    confidence_results.append("Low (Fallback)")

        result_df['predicted_label'] = label_results
        result_df['confidence_level'] = confidence_results
        return result_df
