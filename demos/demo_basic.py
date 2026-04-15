import os
import pandas as pd
import time
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
    print("\n--- Phase 1: Label Discovery (Simple) ---")
    suggested_labels = labeler.suggest_labels(df, context=context, n_labels=3)
    print(f"Suggested Labels: {suggested_labels}")
    
    # 4. Smart Discovery Demo (New)
    print("\n--- Phase 1.5: Smart Discovery (Parallel) ---")
    print(f"Sampling 3 chunks of 5 records each...")
    from auto_labeler.strategies import ParallelDiscoveryStrategy
    parallel_strategy = ParallelDiscoveryStrategy(labeler.llm, num_samples=3, sample_size=5)
    
    smart_labels = labeler.suggest_labels(df, context, n_labels=3, strategy=parallel_strategy)
    print(f"Smart Suggested Labels (Merged): {smart_labels}")
    
    # Use smart labels effectively if they exist
    if smart_labels:
        suggested_labels = list(set(suggested_labels + smart_labels))
    
    # 4. Assignment
    print("\n--- Phase 2: Label Assignment ---")
    time.sleep(10) # Avoid Rate Limits
    labeled_df = labeler.label_dataset(df, labels=suggested_labels, context=context)
    
    print("\nResults:")
    print(labeled_df[["text", "label"]])
    
    # 5. Domain Knowledge (Few-Shot) Demo
    print("\n--- Phase 2.5: Domain Knowledge (Few-Shot) ---")
    # Let's say we have an ambiguous case or we want to enforce a rule.
    # Example: "Reset password" might be "Support" but we want "Account Security".
    # And we want to see if an example fixes it.
    
    # Creating a tricky case
    tricky_df = pd.DataFrame({"text": ["I forgot my password again."]})
    print(f"Labeling tricky input: {tricky_df['text'].iloc[0]}")
    
    # Without examples (might go to General Support)
    res_no_ex = labeler.label_dataset(tricky_df, labels=suggested_labels + ["Account Security"], context=context)
    print(f"Without Example: {res_no_ex['label'].iloc[0]}")
    
    # With Example
    examples = [
        {"text": "How do I reset my password?", "label": "Account Security"}
    ]
    res_with_ex = labeler.label_dataset(
        tricky_df, 
        labels=suggested_labels + ["Account Security"], 
        context=context,
        examples=examples
    )
    print(f"With Example: {res_with_ex['label'].iloc[0]}")

    # 5. Consensus Mode Demo
    print("\n--- Phase 3: Consensus & Confidence ---")
    time.sleep(10) # Avoid Rate Limits
    print("Running with 3 judges (Gemini Flash)...")
    
    from auto_labeler.strategies import ConsensusLabelingStrategy
    
    models_list = [
        "gemini/gemini-2.5-flash", 
        "gemini/gemini-2.5-flash", 
        "gemini/gemini-2.5-flash"
    ]
    
    # Create api_keys dict for all models using the single key we have
    api_keys_map = {m: api_key for m in models_list}
    api_keys_map["gemini/gemini-1.5-pro"] = api_key

    consensus_strategy = ConsensusLabelingStrategy(
        models=models_list,
        adjudicator_model="gemini/gemini-1.5-pro",
        api_keys=api_keys_map
    )
    
    consensus_df = labeler.label_dataset(
        df, 
        labels=suggested_labels, 
        context=context,
        strategy=consensus_strategy
    )
    
    print("\nConsensus Results:")
    print(consensus_df[["text", "label", "confidence_level"]])

    # 6. Final Usage Check
    print("\n--- Final Session Stats ---")
    usage = labeler.get_usage()
    print(f"Total Tokens: {usage['total_tokens']}")
    print(f"Estimated Cost: ${usage['total_cost_usd']:.4f}")

    # Cleanup
    if os.path.exists(dummy_csv):
        os.remove(dummy_csv)

if __name__ == "__main__":
    main()
