# Auto-Labeler 🏷️

**Unlabeled data to insights in two steps.**

Auto-Labeler is a pragmatic AI tool for developers who need to categorize text data at scale without manually labeling thousands of rows. It uses state-of-the-art LLMs (Gemini, OpenAI, Anthropic) to **discover** your taxonomy and then **assign** labels with production-grade reliability.

## Why Auto-Labeler?

*   **🔍 No Labels? No Problem**: Automatically discovers the latent structure of your data and suggests a taxonomy.
*   **⚡ Built for Scale**: Asynchronous batching means you can process thousands of records efficiently.
*   **💾 Zero-Cost Re-runs**: Persistent disk caching ensures you never pay for the same completion twice.
*   **💰 Transparent Costs**: Real-time USD telemetry for every session.
*   **🛡️ Production-Ready**: Strict Pydantic validation and retry logic built-in.

## The Pragmatic Workflow

Most labeling tasks follow a simple pattern: Discover -> Label.

```python
from auto_labeler import AutoLabeler
import pandas as pd

# 1. Initialize (Defaults to Gemini Flash - Fast & Cheap)
labeler = AutoLabeler()

# 2. Discover: "I don't know what my categories are"
df = pd.read_csv("unlabeled_feedback.csv")
suggested_labels = labeler.suggest_labels(
    df, 
    context="Analyzing customer feedback for a mobile app"
)
print(f"Top discovered categories: {suggested_labels}")

# 3. Label: "Now apply those categories to everything"
results_df = labeler.label_dataset(
    df,
    labels=suggested_labels,
    context="Customer Support Ticketing"
)

# 4. Check efficiency
print(labeler.get_usage()) 
# -> {'total_tokens': 1540, 'total_cost_usd': 0.0004, ...}
```

## Installation

```bash
pip install auto-labeler-ai
```

---

<div align="center">
  <p>Stop labeling. Start analyzing.</p>
</div>
