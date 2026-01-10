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

## Features & Documentation
- **[Advanced Usage Guide](docs/advanced_usage.md)**: Learn about Consensus, Smart Discovery, and Few-Shot Learning.
- **Labeling Strategies**: Choose between Speed (Simple) or Accuracy (Consensus).
- **Smart Discovery**: Automatically find edge-case labels using parallel sampling.
- **Domain Knowledge**: Teach the AI with "Few-Shot" examples for higher accuracy.
- **Flexible Backend**: Works with OpenAI, Gemini, Anthropic, and more via LiteLLM.
- **Pandas Integration**: Native dataframe support.

## Installation
