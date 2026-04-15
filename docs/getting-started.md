# Getting Started

Getting up and running with Auto-Labeler is simple. Follow this guide to set up your environment and run your first labeling job.

## 1. Installation

Auto-Labeler requires Python 3.9 or higher.

```bash
pip install auto-labeler
```

## 2. Setting up API Keys

Auto-Labeler uses **LiteLLM** under the hood, supporting 100+ LLM providers. You'll need an API key for the model you want to use.

Create a `.env` file in your project root or set the environment variables in your shell:

### Google Gemini (Recommended)
```bash
export GEMINI_API_KEY="your-api-key-here"
```

### OpenAI
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## 3. Your First Labeling Job

Create a file named `label_data.py`:

```python
import pandas as pd
from auto_labeler import AutoLabeler
from dotenv import load_dotenv

load_dotenv()

# Sample data
data = {
    "text": [
        "How do I reset my password?",
        "I was charged twice for my subscription.",
        "Can I add a second user to my plan?",
    ]
}
df = pd.DataFrame(data)

# Initialize
labeler = AutoLabeler(model_name="gemini/gemini-flash-latest")

# Run Labeling
results = labeler.label_dataset(
    df,
    labels=["Account Access", "Billing", "Feature Request"],
    context="SaaS Platform customer support",
    batch_size=2 # Fast & efficient!
)

print(results[["text", "predicted_label"]])
print(f"Cost: ${labeler.get_usage()['total_cost_usd']}")
```

## 4. Discovering Labels

If you don't know your categories yet, use `suggest_labels`:

```python
suggested = labeler.suggest_labels(
    df,
    context="Analyzing user feedback for a mobile game",
    n_labels=5
)
print(f"Suggested Categories: {suggested}")
```

## 5. Using the CLI

Auto-Labeler also comes with a convenient CLI tool:

```bash
# Discover labels
auto-labeler discover -i feedback.csv -c "Exploring new labels" -n 5

# Apply labels with batching
auto-labeler label -i feedback.csv -l "Bug,Feature,UI" -c "Game Feedback" -o results.csv --batch-size 10
```
