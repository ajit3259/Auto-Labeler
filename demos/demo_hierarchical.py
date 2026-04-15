import os
import sys
import pandas as pd
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from auto_labeler.core import AutoLabeler
from auto_labeler.strategies import HierarchicalLabelingStrategy

def main():
    load_dotenv()
    print("🚀 Initializing Hierarchical Labeling Demo...")
    
    # Use the verified Gemini model for 2026 environment
    labeler = AutoLabeler(model_name="gemini/gemini-flash-latest")

    # Data with distinct categories and sub-themes
    data = {"text": [
        "My credit card was declined at the store.",
        "How do I request a refund for my last order?",
        "The database connection timed out after 30 seconds.",
        "I need to reset my login password.",
        "Why was I charged twice for the same transaction?"
    ]}
    df = pd.DataFrame(data)

    # Define a taxonomy
    taxonomy = {
        "Finance": ["Payment Issue", "Refund Request", "Billing Error"],
        "Technical": ["Database error", "Login Issue", "API Timeout"]
    }
    
    context = "Customer support tickets for a fintech and tech platform."

    print("\n🏗️ Setting up Hierarchical Strategy...")
    strategy = HierarchicalLabelingStrategy(labeler.llm, taxonomy=taxonomy)
    
    print("🏷️ Labeling dataset (Two-Pass)...")
    labeled_df = labeler.label_dataset(
        df, 
        labels=[], # Ignored by strategy
        context=context,
        strategy=strategy
    )
    
    print("\n✅ Results:")
    print(labeled_df[["text", "predicted_category", "predicted_sub_label"]])

if __name__ == "__main__":
    main()
