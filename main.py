import os
from model import load_model, fine_tune_model, generate_stylized_summary
from data import prepare_stylized_dataset, prepare_dataset_for_training
from evaluation import evaluate_summaries
from ui import create_ui
from datasets import concatenate_datasets


def main():
    print("Starting Controlled Text Summarization System")

    # 1. Load base model
    print("Loading base model...")
    model, tokenizer = load_model("facebook/bart-large-cnn")

    # Check if we already have a fine-tuned model
    if os.path.exists("./summarization_model") and os.path.isdir(
        "./summarization_model"
    ):
        print("Loading fine-tuned model...")
        model, tokenizer = load_model("./summarization_model")
    else:
        # 2. Prepare dataset with stylized summaries
        print("Preparing stylized dataset...")
        df = prepare_stylized_dataset()

        # 3. Fine-tune model
        print("Fine-tuning model...")
        # Prepare dataset for all styles
        datasets = {}
        for style in ["formal", "informal", "humorous", "poetic"]:
            print(f"Preparing dataset for {style} style...")
            datasets[style] = prepare_dataset_for_training(df, tokenizer, style)

        # Combine datasets
        combined_dataset = {
            "train": concatenate_datasets(
                [
                    datasets["formal"]["train"],
                    datasets["informal"]["train"],
                    datasets["humorous"]["train"],
                    datasets["poetic"]["train"],
                ]
            ),
            "validation": concatenate_datasets(
                [
                    datasets["formal"]["validation"],
                    datasets["informal"]["validation"],
                    datasets["humorous"]["validation"],
                    datasets["poetic"]["validation"],
                ]
            ),
        }

        # Fine-tune the model
        print("Starting model fine-tuning...")
        model = fine_tune_model(model, tokenizer, combined_dataset)

    # 4. Evaluate the model (optional)
    print("Evaluating model...")
    try:
        df = prepare_stylized_dataset()
        test_texts = df["text"].tolist()[:5]  # Sample for testing
        results = {}
        for style in ["formal", "informal", "humorous", "poetic"]:
            generated_summaries = [
                generate_stylized_summary(text, model, tokenizer, style)
                for text in test_texts
            ]
            reference_summaries = df[f"summary_{style}"].tolist()[:5]
            results[style] = evaluate_summaries(
                generated_summaries, reference_summaries
            )

        print("Evaluation Results:", results)
    except Exception as e:
        print(f"Error during evaluation: {e}")

    # 5. Launch UI
    print("Launching UI...")
    interface = create_ui(model, tokenizer)
    interface.launch()


if __name__ == "__main__":
    main()
