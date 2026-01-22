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
    # API Keys are handled via Environment Variables (OPENAI_API_KEY, GEMINI_API_KEY, etc.)
    # Or pass explicit keys: api_keys={"gpt-4o": "...", ...}
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

## 4. Iterative Discovery (Advanced)
A powerful "Human-in-the-Loop" style agentic strategy. Instead of just guessing labels once, it actively looks for edge cases using a "Generate -> Classify -> Refine" loop.

It supports three modes for different needs:

### Mode: `refine` (Default)
Best for **Deep Discovery**. It finds "Edge Cases" that simple sampling misses.
1.  **Seed**: Finds initial labels from a small sample.
2.  **Sweep**: Tries to classify a large batch using those labels. If items don't fit, they are marked as "Other".
3.  **Refine**: Runs discovery *only* on the "Other" items to find missing labels.

```python
from auto_labeler.strategies import IterativeDiscoveryStrategy

# Find initial labels, then sweep 100 rows to find edge cases
strategy = IterativeDiscoveryStrategy(
    labeler.llm, 
    mode="refine",
    seed_sample_size=10,
    batch_size=100,
    other_threshold=5 # Refine if 5+ items don't fit
)
labels = labeler.suggest_labels(df, context=context, strategy=strategy)
```

### Mode: `evolve`
Best for **Concept Drift** or **Learning**. It learns sequentially.
1.  Batch 1 -> Learn Labels A.
2.  Batch 2 -> "Here are Labels A. Update them based on this new data." -> Labels B.

```python
strategy = IterativeDiscoveryStrategy(
    labeler.llm,
    mode="evolve",
    batch_size=50 # Learn in chunks of 50
)
```

### Mode: `aggregate`
Best for **Speed/Parallelism**. It learns independently and merges.
1.  Batch 1 -> Labels A.
2.  Batch 2 -> Labels B.
3.  **Merge**: Uses an LLM to semantically merge A and B (e.g., "Login Error" + "Login Fail" -> "Login Issue").

```python
strategy = IterativeDiscoveryStrategy(
    labeler.llm,
    mode="aggregate",
    batch_size=50
)
```

## 5. Embedding Discovery (Geometric)
Finds labels by converting text into "Vectors" and clustering them geometrically. This is excellent for finding **Thematic Groups** in large datasets without relying on random text sampling.

### Method: `K-Means` (Default)
You specify `n_clusters`. Good if you know roughly how many categories you want.

```python
from auto_labeler.strategies import EmbeddingDiscoveryStrategy

strategy = EmbeddingDiscoveryStrategy(
    labeler.llm,
    clustering_method="kmeans",
    n_clusters=10, 
    sample_size=1000 # Use a large sample for better geometry
)
labels = labeler.suggest_labels(df, context=context, strategy=strategy)
```

### Method: `DBSCAN`
Automatically finds the number of clusters based on density. Good for noisy data, but requires tuning `eps`.

```python
strategy = EmbeddingDiscoveryStrategy(
    labeler.llm,
    clustering_method="dbscan",
    eps=0.5,
    min_samples=5
)
```
