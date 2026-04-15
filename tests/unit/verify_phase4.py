import asyncio
import pandas as pd
from auto_labeler.core import AutoLabeler
from auto_labeler.strategies import SimpleLabelingStrategy
from dotenv import load_dotenv

async def test_batching_and_async():
    load_dotenv()
    print("🚀 Initializing Phase 4 Verification...")
    
    labeler = AutoLabeler(model_name="gemini/gemini-flash-latest")
    
    df = pd.DataFrame({"text": [
        "I need help with my billing.",
        "The app crashed on my iPhone.",
        "How do I change my subscription?",
        "Feature request: Dark mode.",
        "I can't log in to my account."
    ]})
    
    context = "Customer support tickets."
    labels = ["Billing", "Technical Issue", "Feature Request", "Account Access"]
    
    # 1. Test Batched Labeling (Sync)
    print("\n📦 Testing Batched Labeling (Batch Size: 3)...")
    strategy_batch = SimpleLabelingStrategy(labeler.llm, batch_size=3)
    res_batch = labeler.label_dataset(df, labels=labels, context=context, strategy=strategy_batch)
    print("Batch Results:")
    print(res_batch[["text", "predicted_label"]])
    
    # 2. Test Async Labeling
    print("\n⚡ Testing Async Labeling (All at once)...")
    res_async = await strategy_batch.alabel(df, labels=labels, context=context, prompts_dir=labeler.prompts_dir)
    print("Async Results:")
    print(res_async[["text", "predicted_label"]])
    
    # 3. Verify Telemetry
    usage = labeler.get_usage()
    print("\n📊 Telemetry Summary:")
    print(f"Total Tokens: {usage['total_tokens']}")
    print(f"Prompt Tokens: {usage['prompt_tokens']}")
    print(f"Completion Tokens: {usage['completion_tokens']}")
    
    if usage['total_tokens'] > 0:
        print("✅ Telemetry verified.")
    else:
        print("❌ Telemetry failed (0 tokens).")

if __name__ == "__main__":
    asyncio.run(test_batching_and_async())
