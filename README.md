# Auto-Labeler

An AI-powered data labeling library for Python.

## Features
- **Discovery**: Automatically suggests labels/taxonomy from your data.
- **Assignment**: labels your dataset using LLMs (via LiteLLM).
- **Flexible**: Works with CSVs and Pandas DataFrames.

## Installation
```bash
pip install auto-labeler
```

## Supported Models
We use [LiteLLM](https://docs.litellm.ai/docs/providers) under the hood, so you can use almost any LLM provider.
Pass the model name in the `model_name` argument (e.g., `gpt-4o`, `gemini/gemini-1.5-flash`, `claude-3-opus`).

See the [LiteLLM Providers Docs](https://docs.litellm.ai/docs/providers) for the exact string format for your provider.
