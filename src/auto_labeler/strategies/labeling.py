from typing import List, Protocol, Optional, Union, Dict
import pandas as pd
from ..llm import LLMAdapter
import pathlib
import yaml
import asyncio
from jinja2 import Template

class LabelingStrategy(Protocol):
    """
    Protocol definition for labeling strategies.
    Strategies define how labels are assigned to a dataframe given a context and set of labels.
    """
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
        """
        Apply labels to the dataframe.
        
        Args:
            df: Input dataframe containing the text to label.
            labels: List of allowed labels.
            context: Description of the dataset context.
            prompts_dir: Directory containing YAML prompt templates.
            target_column: Name of the column containing text to label.
            multi_label: Whether multiple labels can be assigned to a single record.
            examples: Optional list of few-shot examples (dictionaries with 'text' and 'label').
            
        Returns:
            DataFrame with 'predicted_label' column added.
        """
        ...

class SimpleLabelingStrategy:
    """
    A cost-effective strategy that uses a single LLM call per record.
    Best for simple tasks or initial passes.
    """
    def __init__(self, llm: LLMAdapter, batch_size: int = 1):
        """
        Args:
            llm: The LLM adapter to use.
            batch_size: Number of records to process in a single LLM call. Default is 1.
        """
        self.llm = llm
        self.batch_size = batch_size

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
        if self.batch_size > 1:
            return self._label_batched(df, labels, context, prompts_dir, target_column, multi_label, examples)
        
        result_df = df.copy()
        label_results = []
        prompt_template = self._load_prompt(prompts_dir, "assignment")
        
        multi_label_instruction = 'Select strictly one label.' if not multi_label else 'Select one or more labels.'
        output_format_instruction = 'a string' if not multi_label else 'a list of strings'
        
        template = Template(prompt_template)
        
        for index, row in result_df.iterrows():
            record_content = row[target_column] if target_column in row else str(row.to_dict())
            
            prompt = template.render(
                context=context,
                labels=labels,
                record_content=record_content,
                multi_label_instruction=multi_label_instruction,
                output_format_instruction=output_format_instruction,
                examples=examples
            )
            
            try:
                schema = {
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string" if not multi_label else "array",
                            "items": {"type": "string"} if multi_label else None,
                            "description": "The predicted label(s) for the content."
                        }
                    },
                    "required": ["label"]
                }
                response = self.llm.generate_structured(prompt, response_schema=schema)
                assigned_label = response.get("label")
                
                if not multi_label and isinstance(assigned_label, list):
                    assigned_label = assigned_label[0] if assigned_label else None
                
                label_results.append(assigned_label)
            except Exception as e:
                print(f"Error in simple labeling: {e}")
                label_results.append(None)
        
        result_df['predicted_label'] = label_results
        return result_df

    def _label_batched(
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
        label_results = [None] * len(df)
        prompt_template = self._load_prompt(prompts_dir, "batch_assignment")
        template = Template(prompt_template)
        
        multi_label_instruction = 'Select strictly one label per record.' if not multi_label else 'Select one or more labels per record.'
        
        # Chunk the dataframe
        chunks = [df[i:i + self.batch_size] for i in range(0, len(df), self.batch_size)]
        
        for chunk_idx, chunk in enumerate(chunks):
            # Formulate items with IDs relative to the original dataframe
            items = []
            for i, (idx, row) in enumerate(chunk.iterrows()):
                content = row[target_column] if target_column in row else str(row.to_dict())
                items.append({"id": idx, "text": content})
            
            prompt = template.render(
                context=context,
                labels=labels,
                items=items,
                multi_label_instruction=multi_label_instruction,
                examples=examples
            )
            
            try:
                schema = {
                    "type": "object",
                    "properties": {
                        "results": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "integer"},
                                    "label": {"type": "string" if not multi_label else "array"}
                                },
                                "required": ["id", "label"]
                            }
                        }
                    },
                    "required": ["results"]
                }
                response = self.llm.generate_structured(prompt, response_schema=schema)
                results = response.get("results", [])
                
                # Map results back to the original index
                for res in results:
                    rid = res.get("id")
                    label = res.get("label")
                    if rid is not None and rid in df.index:
                        # Find position of index in original result_df
                        pos = df.index.get_loc(rid)
                        label_results[pos] = label
                        
            except Exception as e:
                print(f"Error in batched labeling chunk {chunk_idx}: {e}")
                
        result_df['predicted_label'] = label_results
        return result_df

    async def alabel(
        self, 
        df: pd.DataFrame, 
        labels: List[str], 
        context: str, 
        prompts_dir: pathlib.Path,
        target_column: str = "text",
        multi_label: bool = False,
        examples: Optional[List[Dict[str, str]]] = None
    ) -> pd.DataFrame:
        """
        Asynchronous labeling with parallelized batch execution.
        """
        result_df = df.copy()
        label_results = [None] * len(df)
        prompt_template = self._load_prompt(prompts_dir, "batch_assignment" if self.batch_size > 1 else "assignment")
        template = Template(prompt_template)
        
        multi_label_instruction = 'Select strictly one label per record.' if not multi_label else 'Select one or more labels per record.'
        
        chunks = [df[i:i + self.batch_size] for i in range(0, len(df), self.batch_size)]
        
        async def process_chunk(chunk):
            items = []
            for idx, row in chunk.iterrows():
                content = row[target_column] if target_column in row else str(row.to_dict())
                items.append({"id": idx, "text": content})
            
            # Use different prompt logic for batch vs single
            if self.batch_size > 1:
                prompt = template.render(context=context, labels=labels, items=items, multi_label_instruction=multi_label_instruction, examples=examples)
                schema = {
                    "type": "object",
                    "properties": {
                        "results": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {"id": {"type": "integer"}, "label": {"type": "string" if not multi_label else "array"}},
                                "required": ["id", "label"]
                            }
                        }
                    },
                    "required": ["results"]
                }
            else:
                # Single item fallback (still wrapped in async)
                item = items[0]
                prompt = template.render(
                    context=context, labels=labels, record_content=item["text"], 
                    multi_label_instruction=multi_label_instruction, 
                    output_format_instruction='a string' if not multi_label else 'a list of strings',
                    examples=examples
                )
                schema = {"type": "object", "properties": {"label": {"type": "string" if not multi_label else "array"}}, "required": ["label"]}

            try:
                response = await self.llm.agenerate_structured(prompt, response_schema=schema)
                if self.batch_size > 1:
                    return response.get("results", [])
                else:
                    return [{"id": items[0]["id"], "label": response.get("label")}]
            except Exception as e:
                print(f"Error in async labeling chunk: {e}")
                return []

        # Run all chunks concurrently
        tasks = [process_chunk(chunk) for chunk in chunks]
        chunk_responses = await asyncio.gather(*tasks)
        
        # Merge all responses
        for response_list in chunk_responses:
            for res in response_list:
                rid = res.get("id")
                label = res.get("label")
                if rid is not None and rid in df.index:
                    pos = df.index.get_loc(rid)
                    label_results[pos] = label
                    
        result_df['predicted_label'] = label_results
        return result_df

class ConsensusLabelingStrategy:
    """
    A high-accuracy strategy that uses multiple models (judges) to vote on labels.
    Disagreements are resolved by a powerful 'Adjudicator' model.
    """
    def __init__(self, models: List[str], adjudicator_model: str, api_keys: Optional[Dict[str, str]] = None):
        """
        Args:
            models: List of model names to use as judges (e.g. ['gpt-3.5-turbo', 'gemini-1.5-flash']).
            adjudicator_model: Strong model (e.g. 'gpt-4o') to decide in case of disagreement.
            api_keys: Optional dictionary mapping model names to API keys. 
                      If None, relies on environment variables (recommended).
                      Example: {'gpt-4o': 'sk-...', 'gemini/gemini-1.5-flash': 'AI...'}
        """
        self.adapters = []
        for m in models:
            # Check if specific key provided, else None (Env Var)
            key = api_keys.get(m) if api_keys else None
            self.adapters.append(LLMAdapter(model_name=m, api_key=key))
            
        adj_key = api_keys.get(adjudicator_model) if api_keys else None
        self.adjudicator = LLMAdapter(model_name=adjudicator_model, api_key=adj_key)

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
        
        # TODO: Implement batch processing for consensus voting as well.
        
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

class HierarchicalLabelingStrategy:
    """
    A strategy for labeling with a Category > Sub-category structure.
    Performs two passes:
    1. Classifies the high-level Category.
    2. Classifies the Sub-category based on the predicted Category.
    """
    def __init__(self, llm: LLMAdapter, taxonomy: Dict[str, List[str]]):
        """
        Args:
            llm: The LLM adapter to use.
            taxonomy: A dictionary mapping categories to lists of sub-categories.
                      Example: {"Finance": ["Payment", "Refund"], "Tech": ["Server", "DB"]}
        """
        self.llm = llm
        self.taxonomy = taxonomy

    def label(
        self, 
        df: pd.DataFrame, 
        labels: List[str], # This will be ignored in favor of taxonomy keys first
        context: str, 
        prompts_dir: pathlib.Path,
        target_column: str = "text",
        multi_label: bool = False,
        examples: Optional[List[Dict[str, str]]] = None
    ) -> pd.DataFrame:
        # Pass 1: Category Labeling
        categories = list(self.taxonomy.keys())
        simple_strategy = SimpleLabelingStrategy(self.llm)
        
        print(f"Pass 1: Identifying Categories ({categories})...")
        df_with_cats = simple_strategy.label(
            df=df,
            labels=categories,
            context=context,
            prompts_dir=prompts_dir,
            target_column=target_column,
            multi_label=False, # Hierarchical category is usually single
            examples=examples
        )
        df_with_cats = df_with_cats.rename(columns={"predicted_label": "predicted_category"})
        
        # Pass 2: Sub-category Labeling
        print("Pass 2: Identifying Sub-categories...")
        sub_results = []
        
        for _, row in df_with_cats.iterrows():
            category = row["predicted_category"]
            if category not in self.taxonomy:
                sub_results.append(None)
                continue
            
            sub_labels = self.taxonomy[category]
            record_content = row[target_column]
            
            # Formulate sub-context
            sub_context = f"{context} This item is confirmed to be in the '{category}' category."
            
            # Localized labeling for this specific record
            temp_df = pd.DataFrame([row])
            res = simple_strategy.label(
                df=temp_df,
                labels=sub_labels,
                context=sub_context,
                prompts_dir=prompts_dir,
                target_column=target_column,
                multi_label=multi_label,
                examples=None # Examples might need to be specific to sub-category, for now skip
            )
            sub_results.append(res["predicted_label"].iloc[0])
            
        df_with_cats["predicted_sub_label"] = sub_results
        # Final combined label (optional, but good for compatibility)
        df_with_cats["predicted_label"] = df_with_cats.apply(
            lambda r: f"{r['predicted_category']} > {r['predicted_sub_label']}" if r['predicted_sub_label'] else r['predicted_category'], 
            axis=1
        )
        
        return df_with_cats
