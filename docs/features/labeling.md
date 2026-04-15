# Labeling Strategies 🏷️

Auto-Labeler supports various strategies to balance cost, accuracy, and depth.

## Strategies

### Simple Labeling
The standard approach. Each record is processed individually or in batches.

### Hierarchical Labeling
Uses a two-pass approach. 
1. Identify the high-level **Category**.
2. Identify the specific **Sub-category** based on a nested taxonomy.

### Consensus Labeling
Calls multiple models (or the same model multiple times) and uses an adjudicator LLM to resolve disagreements. Best for high-precision requirements.