# Choosing a Strategy

Auto-Labeler has two families of strategies: **Discovery** (find what labels to use) and **Labeling** (assign those labels). Both families offer multiple options. This page helps you pick the right one.

---

## Discovery Strategies

Use these when you don't have a predefined taxonomy and want the library to suggest one from your data.

| Strategy | Best for | Speed | Cost |
|---|---|---|---|
| **Simple** | Quick exploration, small datasets (<500 rows) | Fast | Low |
| **Iterative** | Production taxonomy, edge-case coverage | Medium | Medium |
| **Embedding** | Large datasets (1k+ rows), semantic clusters | Slow | Medium |

### Simple Discovery
Samples a random batch of records and asks the LLM to name the top themes in one shot. Good for a first look, but it may miss rare categories if they don't appear in the sample.

```python
suggested = labeler.suggest_labels(df, context="Help desk tickets", n_labels=5)
```

**Use when**: You want a fast sanity-check of what's in your data. Don't use for final taxonomies on imbalanced datasets.

---

### Iterative Discovery ✅ Recommended default
Runs multiple passes: finds an initial taxonomy, then deliberately hunts for records that don't fit ("Other"), and refines the labels. Converges on a taxonomy that covers the full distribution.

```python
from auto_labeler.strategies import IterativeDiscoveryStrategy

strategy = IterativeDiscoveryStrategy(labeler.llm, max_iterations=3, sample_size=50)
suggested = labeler.suggest_labels(df, context="Help desk tickets", strategy=strategy)
```

**Use when**: You want labels you can actually ship. The right choice for most datasets.

---

### Embedding Discovery
Embeds every record (up to `sample_size`), clusters the vectors, then asks the LLM to name each cluster. Finds structure the LLM alone might miss because it reads across the whole distribution, not just a sample.

```python
from auto_labeler.strategies import EmbeddingDiscoveryStrategy

strategy = EmbeddingDiscoveryStrategy(
    labeler.llm,
    clustering_method="kmeans",
    n_clusters=8,
)
suggested = labeler.suggest_labels(df, context="Log analysis", strategy=strategy)
```

**Use when**: Dataset is large (1k+ rows), topics are subtle/overlapping, or you distrust LLM sampling bias. Requires an embedding-capable model.

---

## Labeling Strategies

Use these once you have a fixed list of labels and want to assign them to every row.

| Strategy | Best for | Speed | Cost | Accuracy |
|---|---|---|---|---|
| **Simple** | Most use-cases | Fast | Low | Good |
| **Hierarchical** | Large/nested taxonomies (20+ labels) | Medium | Medium | Good |
| **Consensus** | High-stakes labels, disagreement detection | Slow | High | Highest |

### Simple Labeling ✅ Recommended default
Single LLM call per record (or per batch). Fast and cost-efficient. Handles multi-label. Validates returned labels against your allowed list automatically.

```python
results = labeler.label_dataset(
    df,
    labels=["Billing", "Technical", "General"],
    context="Customer support tickets",
)
print(results[["text", "label"]])
```

**Use when**: You have a clear taxonomy and want results quickly. The right starting point for almost every project.

---

### Hierarchical Labeling
Two-pass approach: first classifies the broad **Category**, then narrows to the **Sub-category** given that category. Keeps each individual prompt small, which improves accuracy when you have 20+ labels.

```python
from auto_labeler.strategies import HierarchicalLabelingStrategy

taxonomy = {
    "Billing":    ["Refund", "Invoice Issue", "Payment Failed"],
    "Technical":  ["App Crash", "Slow Performance", "Login Issue"],
    "General":    ["Feature Request", "Feedback", "Other"],
}

strategy = HierarchicalLabelingStrategy(labeler.llm, taxonomy=taxonomy)
results = labeler.label_dataset(df, labels=taxonomy, context="SaaS support", strategy=strategy)
```

**Use when**: Your taxonomy has more than ~15 labels, or is naturally hierarchical (Category → Sub-category). Using Simple labeling with 30 labels in the prompt degrades accuracy; Hierarchical keeps each decision focused.

---

### Consensus Labeling
Fans out to multiple models (or the same model multiple times), collects votes, and only calls an Adjudicator model when the judges disagree. Returns a `confidence_level` column (`High (Unanimous)`, `Medium (Adjudicated)`, `Low (Fallback)`).

```python
from auto_labeler.strategies import ConsensusLabelingStrategy

strategy = ConsensusLabelingStrategy(
    models=["gemini/gemini-2.5-flash", "openai/gpt-4o-mini"],
    adjudicator_model="gemini/gemini-2.5-pro",
)
results = labeler.label_dataset(
    df,
    labels=["Bug", "Feature Request", "Question"],
    context="GitHub issues",
    strategy=strategy,
)
print(results[["text", "label", "confidence_level"]])
```

**Use when**: Labels drive a business decision (content moderation, medical triage, legal review) and you need to know which rows the models disagreed on. The `confidence_level` column tells you exactly where to invest human review time.

!!! warning "Cost"
    Consensus calls each judge model once per row. With 3 judges + an adjudicator, you pay for ~4 LLM calls per row. Use on a curated high-priority subset, not your full dataset.

---

## Quick Decision Tree

```
Do you have labels already?
├── No  → Discovery
│         ├── Dataset < 500 rows?  → Simple Discovery (fast)
│         ├── Need production-quality coverage?  → Iterative ✅
│         └── Dataset > 1k rows or topics are subtle?  → Embedding
│
└── Yes → Labeling
          ├── < 15 labels?  → Simple ✅
          ├── 15+ labels or nested taxonomy?  → Hierarchical
          └── High-stakes / need confidence scores?  → Consensus
```
