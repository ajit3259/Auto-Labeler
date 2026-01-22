import os
import sys
import pandas as pd
from dotenv import load_dotenv

# Ensure we can import the source code
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from auto_labeler.core import AutoLabeler
from auto_labeler.strategies import IterativeDiscoveryStrategy

def main():
    load_dotenv()
    
    # 1. Setup Labeler
    print("🚀 Initializing Auto-Labeler...")
    labeler = AutoLabeler()

    # 2. Simulate Data (Customer Support/Technical Logs)
    data = {
        "text": [
            "Login failed with error 500",
            "Cannot reset password, link expired",
            "Payment declined on checkout",
            "Cart is empty after refresh",
            "UI is misaligned on mobile",
            "Server timeout during export",
            "API returned 404 on get_user",
            "Background color is wrong in dark mode",
            "Font size too small on homepage",
            "Transaction stuck in pending state"
        ] * 5  # Duplicate to create volume for iteration
    }
    df = pd.DataFrame(data)
    print(f"📄 Loaded {len(df)} records.")

    context = "Technical support logs for a SaaS platform."

    # 3. Use Iterative Discovery (Refine Mode)
    print("\n🔍 Running Iterative Discovery (Mode: Refine)...")
    print("   - This will seed labels, classify a batch, and refine if 'Other' items are found.")
    
    strategy = IterativeDiscoveryStrategy(
        labeler.llm, 
        mode="refine",
        seed_sample_size=5,
        batch_size=20,
        other_threshold=2
    )
    
    discovered_labels = labeler.suggest_labels(df, context=context, strategy=strategy)
    
    print("\n✅ Discovered Labels:")
    for label in discovered_labels:
        print(f"   - {label}")

if __name__ == "__main__":
    main()
