# Auto-Labeler Roadmap & Design

This document captures high-level discussions and design decisions for the future of the library.

## Core Philosophy
- **Pragmatic**: Solve the problem (labelling) with the least friction.
- **Configurable**: Users should choose between "Fast & Cheap" vs. "High Quality & Expensive".

---

## Completed Features

### Phase 1 & 2: Core Engine & Strategies
- **Domain Knowledge (Context)**: Support for Few-Shot examples and detailed problem definitions.
- **Smart Discovery**: Parallel sampling and initial label generation.
- **Consensus & Adjudicator**: Multi-model consensus and disagreement resolution.

### Phase 3: Advanced Intelligence
- ✅ **Iterative Discovery**: Recursive sampling to find edge-case labels.
- ✅ **Embedding-based Clustering**: Using K-Means/DBSCAN to identify semantic groups.
- ✅ **Hierarchical Labeling**: Multi-pass `Category > Sub-category` classification.

### Phase 4: Optimization & Operations
- ✅ **Batch Processing**: High-throughput labeling via grouped records.
- ✅ **Cost Tracking**: Real-time token usage monitoring.
- ✅ **CLI Tool**: `auto-labeler discover` and `auto-labeler label` command-line interfaces.

### Phase 5: Production Ready
- ✅ **Quality Foundation**: Pydantic validation and persistent Disk Caching.
- ✅ **Observability**: Library-wide structured logging.
- ✅ **Packaging**: PyPI-ready `pyproject.toml` metadata.
- ✅ **Documentation**: Beautiful MkDocs material site with API reference.

---

## Future Roadmap (Legacy)

### Phase 6: Active Learning (Proposed)
- **Human-in-the-loop**: Integrated workbench for reviewing low-confidence labels.
- **Auto-Tuning**: Automatically refining prompts based on human corrections.
