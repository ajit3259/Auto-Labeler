Auto-Labeler Roadmap & Design
This document captures high-level discussions and design decisions for the future of the library.

Core Philosophy
Pragmatic: Solve the problem (labelling) with the least friction.
Configurable: Users should choose between "Fast & Cheap" vs. "High Quality & Expensive".
1. Domain Knowledge (Context)
Goal: Move beyond simple string context to a "Knowledge Base" that understands the domain perfectly.

Approach
Iterative/Agentic Knowledge Base:
Instead of pasting a 50-page PDF, the system should allow creating a "Knowledge Base" (likely a vector store or summarized index).
Correction Loop: When humans correct a label in the Workbench (future UI), that correction is fed back into the Knowledge Base.
Immediate Step (V1.5):
Allow users to pass a "Problem Definition" and "Condensed Guidelines/Rules".
Support "Few-Shot Examples": User provides a list of Reference Input->Label pairs.
2. Smart Discovery (Coverage)
Goal: Identify all unique labels in a dataset, not just the common ones found in the first 10 rows.

Strategies
Parallel Sampling (MapReduce):
Sample $N$ batches randomly (e.g., 5 batches of 10 rows).
Send each to LLM to get labels.
Combine unique labels.
Iterative Learning:
Sample Batch 1 -> Labels A.
Sample Batch 2 -> Provide Labels A -> Ask for new missing labels -> Labels B.
Repeat.
Embedding-based Clustering:
Generate embeddings for all rows (cheap model).
Cluster data (K-Means/HDBSCAN).
Send Centroids + Nearby Examples to LLM for labeling.
3. Hierarchical Labeling
Goal: Support taxonomy like Category > Sub-category.

Approach
Discovery: LLMs are smart enough to suggest hierarchy if prompted ("Group these into categories").
Assignment:
Single-pass: "Classify into available Category/Sub-category pairs".
Multi-step (for complex trees): "First Category" -> "Then Sub-category".
4. Confidence via Consensus
Goal: Ensure high quality without asking the human to review everything.

The "Consensus & Adjudicator" Pattern
Instead of relying on a single model's logprobs (which can be overconfident):

Panel of Judges: Send the row to $N$ models (e.g., Gemini Flash, GPT-4o-mini, Claude Haiku).
Consensus Check:
If all agree (3/3) -> High Confidence (Auto-accept).
If disagreement -> Adjudicator LLM (Smarter/Larger model) decides.
Human Fallback: If Adjudicator is unsure, flag for Workbench.
Trade-off: Increases cost/latency. Must be a user-opt-in configuration.

