# Advanced Usage Guide

## Labeling Strategies

### 1. Consensus (Voting)
Improve accuracy by asking multiple models to vote on a label. If they disagree, an "Adjudicator" model makes the final decision.

```python
from auto_labeler.strategies import ConsensusLabelingStrategy

# Define your judges
judges = [
    "gemini/gemini-1.5-flash",
    "gpt-3.5-turbo",
    "claude-3-haiku"
]

strategy = ConsensusLabelingStrategy(
    models=judges,
    adjudicator_model="gpt-4o", 
    api_key=os.environ["GEMINI_API_KEY"]
)

df = labeler.label_dataset(df, labels=labels, context=context, strategy=strategy)
```

## Discovery Strategies

### Smart Discovery (Parallel Sampling)
Find virtually all labels in a large dataset by sampling it multiple times in parallel.

```python
from auto_labeler.strategies import ParallelDiscoveryStrategy

# Sample 5 chunks of 20 records each
strategy = ParallelDiscoveryStrategy(labeler.llm, num_samples=5, sample_size=20)
labels = labeler.suggest_labels(df, context=context, strategy=strategy)
```

## Domain Knowledge (Few-Shot Learning)

### Providing Examples
You can drastically improve accuracy on ambiguous data by providing 1-2 examples of "Input -> Correct Label".

```python
examples = [
    {"text": "I forgot my password.", "label": "Account Security"},
    {"text": "The app is slow.", "label": "Performance"}
]

df = labeler.label_dataset(
    df, 
    labels=labels, 
    context=context, 
    examples=examples
)
```
