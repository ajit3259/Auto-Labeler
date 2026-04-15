# Caching & Telemetry 📊

Auto-Labeler is designed for production transparency and cost-efficiency.

## Disk Caching 💾

By default, Auto-Labeler caches every successful LLM response to a local SQLite database in `.auto_labeler_cache/`.

- **Persistent**: Survives script restarts.
- **Efficient**: Avoids paying for the same prompt twice.
- **Configurable**: You can change the directory or disable it.

## Telemetry & Usage Tracker 💰

Track your consumption in real-time. The `get_usage()` method returns:
- `prompt_tokens`: Input tokens used.
- `completion_tokens`: Output tokens generated.
- `total_tokens`: Sum of everything.
- `total_cost_usd`: Estimated cost in USD (calculated via LiteLLM).

```python
usage = labeler.get_usage()
print(f"Total spent: ${usage['total_cost_usd']}")
```