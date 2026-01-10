# Labeling Strategies

Auto-Labeler supports different strategies to balance cost, speed, and accuracy.

## 1. Simple Strategy (Default)
Fast and cheap. Uses a single LLM call per record.

```python
# Default behavior
labeler.label_dataset(df, labels=labels, context=context)
```

## 2. Consensus Strategy
High accuracy. Queries multiple models (or the same model multiple times) and requires unanimity. If models disagree, an "Adjudicator" model decides.

**Use cases**: Production pipelines, High-stakes data, Creating Golden Datasets.

```python
from auto_labeler.strategies import ConsensusLabelingStrategy

# Define your panel of judges
judges = [
    "gemini/gemini-1.5-flash",
    "gpt-3.5-turbo",
    "claude-3-haiku"
]

strategy = ConsensusLabelingStrategy(
    models=judges,
    adjudicator_model="gpt-4o",  # Stronger model for tie-breaking
    api_key=os.environ["GEMINI_API_KEY"]
)

df = labeler.label_dataset(df, labels=labels, context=context, strategy=strategy)
```

## 3. Smart Discovery (Parallel)
Broader coverage. Instead of looking at just the first 10 rows to find labels, this strategy samples the dataset multiple times to find edge cases.

```python
from auto_labeler.strategies import ParallelDiscoveryStrategy

strategy = ParallelDiscoveryStrategy(labeler.llm, num_samples=5, sample_size=10)
labels = labeler.suggest_labels(df, context=context, strategy=strategy)
```
