# Label Discovery 🔍

Discovery is the process of identifying a consistent taxonomy from raw, unlabeled text. Auto-Labeler provides three strategies to handle different dataset sizes and complexities.

## 1. Simple Discovery
Best for small datasets or a quick overview of themes. It samples a batch of records and asks the LLM to provide the top unique categorical labels.

```python
suggested = labeler.suggest_labels(
    df, 
    context="Analyzing help desk tickets",
    n_labels=5
)
```

## 2. Iterative Discovery (Recommended)
A robust multi-pass strategy. It finds an initial taxonomy, then specifically looks for "Other" or edge-case records to refine the labels until the coverage is complete.

```python
from auto_labeler.strategies import IterativeDiscoveryStrategy

strategy = IterativeDiscoveryStrategy(
    labeler.llm,
    max_iterations=3,
    sample_size=10
)

labels = labeler.suggest_labels(df, "Customer Feedback", strategy=strategy)
```

## 3. Embedding Discovery
Uses vector embeddings and K-Means clustering. This identifies semantic clusters across much larger datasets, ensuring that "discovery" isn't biased by just a few random rows.

```python
from auto_labeler.strategies import EmbeddingDiscoveryStrategy

strategy = EmbeddingDiscoveryStrategy(
    labeler.llm,
    clustering_method="kmeans",
    n_clusters=5,
    embedding_model="gemini/gemini-embedding-001"
)

labels = labeler.suggest_labels(df, "Log Analysis", strategy=strategy)
```