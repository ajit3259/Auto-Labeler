# Labeling Strategies 🏷️

Auto-Labeler supports multiple strategies for assigning labels, allowing you to trade-off between speed, cost, and absolute accuracy.

## 1. Simple Labeling
The default strategy. It processes records in batches using a single LLM prompt per record. High performance and cost-efficient.

```python
results = labeler.label_dataset(
    df,
    labels=["Urgent", "General", "Billing"],
    context="Support Tickets",
    batch_size=10
)
```

## 2. Hierarchical Labeling
Best for complex taxonomies (e.g., 20+ labels). It takes a two-pass approach:
1. Identify the high-level **Category**.
2. Identify the specific **Sub-category** based on a nested structure.

```python
from auto_labeler.strategies import HierarchicalLabelingStrategy

taxonomy = {
    "Billing": ["Refund", "Invoice Issue", "Payment Failed"],
    "Technical": ["App Crash", "Slow Performance", "Login Issue"]
}

strategy = HierarchicalLabelingStrategy(labeler.llm)

results = labeler.label_dataset(
    df, 
    labels=taxonomy, 
    context="SaaS Logs", 
    strategy=strategy
)
```

## 3. Consensus Labeling
The most robust mode. It calls multiple models (or uses multiple samples) and resolves disagreements using an Adjudicator LLM.

```python
from auto_labeler.strategies import ConsensusLabelingStrategy

strategy = ConsensusLabelingStrategy(
    models=["gemini/gemini-2.5-flash", "openai/gpt-4o-mini"],
    adjudicator_model="gemini/gemini-1.5-pro"
)

results = labeler.label_dataset(
    df, 
    labels=["Bug", "Feature"], 
    context="Github Issues", 
    strategy=strategy
)
```