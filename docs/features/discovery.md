# Label Discovery 🔍

If you have a large dataset and aren't sure which categories/labels to use, Auto-Labeler can help you "discover" the latent structure of your data.

## Strategies

### Simple Discovery
Samples the top `N` records and asks the LLM to provide a list of categories. Best for initial exploration.

### Iterative Discovery
A multi-pass strategy that finds initial labels, identifies "Other" records, and then refines the taxonomy to include edge cases.

### Embedding Discovery
Uses vector embeddings and K-Means clustering to identify common themes across a large dataset (e.g., 500+ records).