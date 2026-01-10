import os
import pandas as pd
from auto_labeler.core import AutoLabeler
from auto_labeler.utils import load_data

def main():
    # 1. Setup
    # Ensure GEMINI_API_KEY is set
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Please set your API key in the environment (e.g., export GEMINI_API_KEY='...')")
        return

    print("Initializing AutoLabeler...")
    # Using Gemini via LiteLLM
    labeler = AutoLabeler(model_name="gemini/gemini-2.5-flash", api_key=api_key)

    # 2. Load Data (Creating dummy data for demo purposes)
    print("Creating sample data...")
    dummy_csv = "demo_data.csv"
    pd.DataFrame({
        "text": [
            "The login button is broken on mobile.",
            "I love the new dark mode feature!",
            "How do I reset my password?",
            "The app crashes when I upload a photo.",
            "Great customer service, thanks!"
        ]
    }).to_csv(dummy_csv, index=False)
    
    context = "Customer feedback for a mobile application."
    
    df, _ = load_data(dummy_csv)
    print(f"Loaded {len(df)} records.")

    # 3. Discovery (Optional)
    print("\n--- Phase 1: Label Discovery ---")
    suggested_labels = labeler.suggest_labels(df, context=context, n_labels=3)
    print(f"Suggested Labels: {suggested_labels}")
    
    # 4. Assignment
    print("\n--- Phase 2: Label Assignment ---")
    labeled_df = labeler.label_dataset(df, labels=suggested_labels, context=context)
    
    print("\nResults:")
    print(labeled_df[["text", "predicted_label"]])

    # Cleanup
    if os.path.exists(dummy_csv):
        os.remove(dummy_csv)

if __name__ == "__main__":
    main()
