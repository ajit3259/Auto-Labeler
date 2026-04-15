# Caching & Telemetry 📊

Auto-Labeler is designed to be a "Zero-Waste" library. It proactively tracks usage and avoids unnecessary API calls through aggressive disk caching.

## Persistent Disk Caching 💾

By default, every successful completion is saved to a local SQLite database in `.auto_labeler_cache/`.

### Why it matters
- **Resiliency**: If your script crashes at row 500, rerunning it will "instantly" skip the first 500 rows at zero cost.
- **Cost Savings**: If you decide to add a new column to your dataframe and rerun the labeling, unchanged rows are served from the cache.
- **Speed**: Cache hits resolve in milliseconds (local IO) vs seconds (API network).

```python
# Caching is ON by default
labeler = AutoLabeler()

# To disable or customize:
labeler = AutoLabeler(
    use_cache=False, 
    cache_dir="project_cache/"
)
```

## Financial Telemetry 💰

Track your consumption in real-time. The `get_usage()` method returns a dictionary with raw tokens and estimated USD costs calculated via LiteLLM.

```python
# Run your job
labeler.label_dataset(...)

# Extract stats
usage = labeler.get_usage()

print(f"Total Tokens: {usage['total_tokens']}")
print(f"Estimated Spend: ${usage['total_cost_usd']:.4f}")

# Example of a usage-aware loop
if usage['total_cost_usd'] > 10.0:
    print("WARNING: Reached budget limit!")
```

## Structured Logging 🪵

Auto-Labeler uses the standard Python `logging` module. You can toggle levels to see exactly what is happening under the hood.

```python
labeler = AutoLabeler(log_level="DEBUG")
# Outputs: 
# DEBUG: Initializing LLM search for 'gemini-flash'
# DEBUG: Cache HIT for 'Analyze feedback...'
# INFO: Processed 50/100 records.
```