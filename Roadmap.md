# Auto-Labeler Roadmap & Design

This document captures high-level discussions and design decisions for the future of the library.

## Core Philosophy
- **Pragmatic**: Solve the problem (labelling) with the least friction.
- **Configurable**: Users should choose between "Fast & Cheap" vs. "High Quality & Expensive".

---

## Completed Features (Phase 1 & Phase 2)

### 1. Domain Knowledge (Context)
**Goal**: Move beyond simple string context to a "Knowledge Base" that understands the domain perfectly.

**Approach**:
-   **Immediate Step (V1.5)**:
    -   Allow users to pass a "Problem Definition" and "Condensed Guidelines/Rules".
    -   Support "Few-Shot Examples": User provides a list of Reference Input->Label pairs.

### 2. Smart Discovery (Coverage)
**Goal**: Identify all unique labels in a dataset, not just the common ones found in the first 10 rows.

**Strategies Implemented**:
-   **Parallel Sampling (MapReduce)**:
    -   Sample $N$ batches randomly (e.g., 5 batches of 10 rows).
    -   Send each to LLM to get labels.
    -   Combine unique labels.

### 3. Confidence via Consensus
**Goal**: Ensure high quality without asking the human to review everything.

**The "Consensus & Adjudicator" Pattern**:
1.  **Panel of Judges**: Send the row to $N$ models (e.g., Gemini Flash, GPT-4o-mini, Claude Haiku).
2.  **Consensus Check**:
    -   If all agree (3/3) -> **High Confidence** (Auto-accept).
    -   If disagreement -> **Adjudicator LLM** (Smarter/Larger model) decides.
3.  **Human Fallback**: If Adjudicator is unsure, flag for Workbench.

---

## Future Roadmap


### Phase 3: Advanced Intelligence
**Goal**: Deepen the "understanding" capabilities using embeddings and recursive logic.

**Features**:
1.  **Iterative Discovery**:
    -   Feedback loop: Sample -> Label -> Feed labels back to find *missing* ones.
2.  **Embedding-based Clustering**:
    -   Use `sentence-transformers` to cluster data.
    -   Label only the centroids and outliers to cover the semantic space efficiently.
3.  **Hierarchical Labeling**:
    -   Support `Category > Sub-category` taxonomy.
    -   Two-pass approach: Classify Category first, then Sub-category.

### Phase 4: Optimization & Operations
**Goal**: Make the library production-ready, widely accessible, and cost-efficient.

**Features**:
1.  **Batch Processing**:
    -   Reduce HTTP overhead by sending $N$ records in a single prompt.
    -   Challenge: Parsing multiple JSON outputs reliably.
2.  **Cost Tracking**:
    -   Estimate token usage (Input/Output) per run.
3.  **CLI Tool**:
    -   `auto-labeler run my_data.csv --labels="A,B,C"`
    -   Interactive mode for setup.

### Phase 5: Production Ready
**Goal**: Package the library for public distribution and reliable testing.

**Features**:
1.  **PyPI Package**:
    -   `pip install auto-labeler` via PyPI.
    -   Clean `pyproject.toml` configuration.
2.  **CI/CD Pipeline**:
    -   GitHub Actions for tests and linting.
3.  **Documentation Site**:
    -   MkDocs or Sphinx site hosted on GitHub Pages.
