import click
import pandas as pd
import os
from .core import AutoLabeler
from dotenv import load_dotenv

@click.group()
def main():
    """Auto-Labeler: A pragmatic AI-powered data labeling library."""
    load_dotenv()

@main.command()
@click.option('--input', '-i', required=True, type=click.Path(exists=True), help='Path to input CSV or Parquet file.')
@click.option('--context', '-c', required=True, help='Description of the dataset context.')
@click.option('--n-labels', '-n', default=5, help='Number of labels to suggest.')
@click.option('--model', '-m', default='gemini/gemini-2.5-flash', help='LLM model to use.')
@click.option('--output', '-o', type=click.Path(), help='Path to save discovered labels (YAML).')
def discover(input, context, n_labels, model, output):
    """Suggest potential labels for a dataset."""
    click.echo(f"🔍 Analyzing {input} using {model}...")
    
    # Load data
    if input.endswith('.csv'):
        df = pd.read_csv(input)
    elif input.endswith('.parquet'):
        df = pd.read_parquet(input)
    else:
        click.echo("❌ Unsupported file format. Use .csv or .parquet")
        return

    labeler = AutoLabeler(model_name=model)
    suggested_labels = labeler.suggest_labels(df, context=context, n_labels=n_labels)
    
    click.echo(f"\n✅ Suggested Labels: {', '.join(suggested_labels)}")
    
    if output:
        import yaml
        with open(output, 'w') as f:
            yaml.dump({"labels": suggested_labels, "context": context}, f)
        click.echo(f"💾 Results saved to {output}")

@main.command()
@click.option('--input', '-i', required=True, type=click.Path(exists=True), help='Path to input CSV or Parquet file.')
@click.option('--labels', '-l', required=True, help='Comma-separated list of labels or path to labels YAML.')
@click.option('--context', '-c', required=True, help='Description of the dataset context.')
@click.option('--model', '-m', default='gemini/gemini-2.5-flash', help='LLM model to use.')
@click.option('--output', '-o', required=True, type=click.Path(), help='Path to save labeled output.')
@click.option('--batch-size', '-b', default=1, help='Number of records per LLM call.')
@click.option('--target-column', '-t', default='text', help='Column containing text to label.')
@click.option('--multi-label', is_flag=True, help='Allow multiple labels per record.')
def label(input, labels, context, model, output, batch_size, target_column, multi_label):
    """Apply labels to a dataset."""
    click.echo(f"🏷️ Labeling {input} using {model} (Batch Size: {batch_size})...")
    
    # Load data
    if input.endswith('.csv'):
        df = pd.read_csv(input)
    elif input.endswith('.parquet'):
        df = pd.read_parquet(input)
    else:
        click.echo("❌ Unsupported file format.")
        return

    # Parse labels
    if os.path.exists(labels):
        import yaml
        with open(labels, 'r') as f:
            labels_data = yaml.safe_load(f)
            label_list = labels_data.get('labels', [])
    else:
        label_list = [l.strip() for l in labels.split(',')]

    labeler = AutoLabeler(model_name=model)
    
    # Set up strategy with batching
    from .strategies import SimpleLabelingStrategy
    strategy = SimpleLabelingStrategy(labeler.llm, batch_size=batch_size)
    
    labeled_df = labeler.label_dataset(
        df, 
        labels=label_list, 
        context=context, 
        target_column=target_column,
        multi_label=multi_label,
        strategy=strategy
    )
    
    # Save output
    if output.endswith('.csv'):
        labeled_df.to_csv(output, index=False)
    elif output.endswith('.parquet'):
        labeled_df.to_parquet(output, index=False)
    
    click.echo(f"✅ Labeling complete. Results saved to {output}")
    
    # Show usage stats
    usage = labeler.get_usage()
    click.echo(f"📊 Token Usage: {usage['total_tokens']} tokens used.")

if __name__ == "__main__":
    main()
