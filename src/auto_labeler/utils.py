import pandas as pd
from typing import Optional, Tuple, List, Union

def resolve_label(
    predicted: Union[str, None],
    allowed: List[str],
) -> Union[str, None]:
    """
    Validates and normalises a label returned by the LLM against the allowed list.

    Matching order:
    1. Exact match — returned as-is.
    2. Case-insensitive match — returns the correctly-cased version from allowed.
    3. No match — returns None (caller should log a warning).
    """
    if predicted is None:
        return None
    if predicted in allowed:
        return predicted
    lower_map = {label.lower(): label for label in allowed}
    if predicted.lower() in lower_map:
        return lower_map[predicted.lower()]
    return None


def load_data(
    csv_path: str, 
    context_path: Optional[str] = None, 
    data_dictionary_path: Optional[str] = None
) -> Tuple[pd.DataFrame, str]:
    """
    Loads the CSV data and constructs a context string from provided files.
    
    Args:
        csv_path: Path to the CSV file.
        context_path: Optional path to a text/md file containing domain context.
        data_dictionary_path: Optional path to a file describing the columns.
        
    Returns:
        Tuple containing (DataFrame, Combined Context String)
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Failed to load CSV from {csv_path}: {e}")
    
    context_parts = []
    
    if context_path:
        try:
            with open(context_path, 'r') as f:
                context_parts.append(f"Domain Context:\n{f.read()}")
        except Exception as e:
            print(f"Warning: Could not read context file {context_path}: {e}")
            
    if data_dictionary_path:
        try:
            with open(data_dictionary_path, 'r') as f:
                context_parts.append(f"Data Dictionary/Schema:\n{f.read()}")
        except Exception as e:
            print(f"Warning: Could not read data dictionary {data_dictionary_path}: {e}")
            
    combined_context = "\n\n".join(context_parts)
    return df, combined_context
