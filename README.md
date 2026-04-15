# Auto-Labeler

An AI-powered data labeling library for Python.

## Key Features
- **🔍 Discovery**: Automatically suggest labels/taxonomy using Iterative or Embedding-based discovery.
- **🏷️ Assignment**: Labels your dataset using LLMs (Gemini, OpenAI, Anthropic, etc.).
- **⚡ Batching & Async**: High-throughput processing for large datasets.
- **💾 Disk Caching**: Save costs and time with local persistence.
- **💰 cost Tracking**: Real-time USD cost estimation for every run.
- **🛡️ Validation**: Pydantic-powered fail-fast checks for your data.

## Installation
```bash
pip install auto-labeler-ai
```

## Quick Start
```python
from auto_labeler import AutoLabeler
import pandas as pd

labeler = AutoLabeler(model_name="gemini/gemini-flash-latest")
results = labeler.label_dataset(
    pd.read_csv("data.csv"),
    labels=["Urgent", "Billing", "General"],
    context="Customer support tickets"
)
print(labeler.get_usage())
```

## Documentation
For full guides on Advanced discovery, Caching, and API reference, visit our:
👉 **[Documentation Site](https://ajit3259.github.io/Auto-Labeler)**

## Testing
```bash
pytest
```

## License
MIT

