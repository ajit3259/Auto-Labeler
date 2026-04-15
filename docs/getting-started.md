# Getting Started

This guide walks you through the "Zero to Labeled" workflow. We'll take a raw CSV and turn it into a categorized dataset with cost telemetry.

## 1. Environment Setup

Auto-Labeler is a lightweight wrapper around LiteLLM and Pandas. Ensure you have an API key for your preferred provider (Gemini is the default).

```bash
pip install auto-labeler-ai
export GEMINI_API_KEY="your-key-here"
```

## 2. The Core Workflow: From Zero to Labeled

In a real project, you usually start with text and no taxonomy. Auto-Labeler handles the transition from discovery to classification in one object.

### The "All-in-One" Script

```python
import pandas as pd
from auto_labeler import AutoLabeler

# 1. Load noisy, unlabeled data
df = pd.read_csv("support_tickets.csv")

# 2. Initialize (Default model is gemini-flash for cost-efficiency)
labeler = AutoLabeler()

# 3. Discovery: Let the AI find the themes
# This samples the data and suggests the top N labels
suggested = labeler.suggest_labels(
    df, 
    context="Analyzing help desk tickets for a fintech company",
    n_labels=5
)
print(f"Top themes found: {suggested}")

# 4. Labeling: Apply those themes with production caching
# Batching is handled automatically to reduce Latency/Cost
results = labeler.label_dataset(
    df,
    labels=suggested,
    context="Categorizing tickets for routing to specialist teams",
    batch_size=10 
)

# 5. Review Usage
usage = labeler.get_usage()
print(f"Billed Tokens: {usage['total_tokens']}")
print(f"Estimated Cost: ${usage['total_cost_usd']:.4f}")

# 6. Save results
results.to_csv("labeled_tickets.csv", index=False)
```

## 3. Production Configuration

For real-world use, you may want to customize how the library behaves:

```python
labeler = AutoLabeler(
    model_name="openai/gpt-4o-mini", # Switch providers easily
    use_cache=True,                  # Enable/Disable disk caching
    cache_dir=".my_cache",           # Custom cache location
    log_level="DEBUG"                # See raw prompts and cache hits
)
```

## 4. Next Steps
- **[Labeling Strategies](features/labeling.md)**: Move beyond simple labeling to Hierarchical or Consensus modes.
- **[Discovery Strategies](features/discovery.md)**: Use Embeddings for deep semantic discovery.
- **[Caching & Telemetry](features/telemetry.md)**: Understand how to manage your data and costs.
