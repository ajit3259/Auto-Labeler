# Welcome to Auto-Labeler 🏷️

**Auto-Labeler** is a pragmatic, AI-powered Python library designed to label unlabeled datasets with minimal effort and maximum reliability. 

Whether you have thousands of customer support tickets, product reviews, or medical records, Auto-Labeler helps you go from raw text to structured categories in minutes using state-of-the-art LLMs (Gemini, OpenAI, Anthropic, etc.).

## Key Features

*   **🔍 Automatic Discovery**: Don't know your labels yet? Auto-Labeler can analyze your data and suggest a taxonomy for you.
*   **⚡ High Performance**: Built-in batching and asynchronous support for high-throughput labeling.
*   **💾 Smart Caching**: Save costs and time with local disk caching of LLM responses.
*   **📊 Transparency**: Built-in telemetry tracks every token and estimated USD cost.
*   **🛡️ Robustness**: Pydantic-powered validation and retry logic ensure reliable operation.

## Quick Start

```python
from auto_labeler import AutoLabeler
import pandas as pd

# 1. Load your data
df = pd.read_csv("data.csv")

# 2. Initialize the labeler
labeler = AutoLabeler(model_name="gemini/gemini-flash-latest")

# 3. Label your dataset
labeled_df = labeler.label_dataset(
    df,
    labels=["Urgent", "General Query", "Billing"],
    context="Customer support tickets for a fintech app",
    batch_size=5
)

# 4. Check results and usage
print(labeled_df.head())
print(labeler.get_usage())
```

## Installation

```bash
pip install auto-labeler
```

---

<div align="center">
  <p>Built with ❤️ for pragmatic AI developers.</p>
</div>
