import os
import sys
import pandas as pd
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from auto_labeler.core import AutoLabeler
from auto_labeler.strategies.embedding import EmbeddingDiscoveryStrategy

def main():
    load_dotenv()
    print("🚀 Initializing...")
    labeler = AutoLabeler(model_name="gemini/gemini-2.5-flash")

    # Data with 2 distinct themes
    data = {"text": [
        # Finance
        "Payment failed", "Transaction declined", "Credit card expired", "Bank reject",
        # Tech
        "Server timeout", "Database locked", "500 error", "Connection refused"
    ] * 5}
    df = pd.DataFrame(data)

    print("\n🔍 Running Embedding Discovery (K-Means)...")
    strategy = EmbeddingDiscoveryStrategy(
        labeler.llm,
        clustering_method="kmeans",
        n_clusters=2,
        embedding_model="gemini/gemini-embedding-001",
        text_column="text"
    )
    
    labels = labeler.suggest_labels(df, "Logs", strategy=strategy)
    print("✅ Discovered:", labels)
    
    print("\n--- Usage Stats ---")
    usage = labeler.get_usage()
    print(f"Total Tokens: {usage['total_tokens']}")
    print(f"Estimated Cost: ${usage['total_cost_usd']:.4f}")

if __name__ == "__main__":
    main()
