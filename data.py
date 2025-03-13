import pandas as pd
from datasets import Dataset
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def load_articles():
    """
    Load articles for summarization
    Replace this with your actual data loading logic
    """
    # This is a placeholder - replace with actual data loading
    sample_articles = [
        "Scientists have discovered a new species of deep-sea fish that can withstand extreme pressure. The fish, found at depths of over 8,000 meters, has unique adaptations including specialized cell membranes and pressure-resistant proteins. This discovery may lead to new applications in biotechnology and materials science.",
        "The city council voted yesterday to approve the new urban development plan. The plan includes affordable housing initiatives, expanded public transportation, and investments in green spaces. Critics argue that the plan doesn't address existing infrastructure problems, while supporters praise its forward-thinking approach to urban growth.",
    ]
    return sample_articles


def generate_styled_summary(text: str, style: str, model) -> str:
    """
    Generates a summary of the given text in the specified style using Gemini.

    Args:
        text: The text to summarize.
        style: The desired style (e.g., "formal", "informal").
        model: The initialized Gemini model.

    Returns:
        The generated summary, or an empty string if an error occurred.
    """
    if not text:
        return ""  # Handle empty input

    prompt = f"Summarize the following text in a {style} style:\n\n{text}"

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=150,
                temperature=0.7,  # Add temperature for some variability
                top_p=0.95,  # Add top_p
                top_k=40,  # Add top_k
            ),
        )
        if response.text:
            return response.text
        else:
            print(
                f"Warning: Empty response for style '{style}' and text: {text[:50]}..."
            )  # Show first 50 chars
            return ""

    except Exception as e:
        print(f"Error generating summary (style: {style}): {e}")
        return ""


def prepare_stylized_dataset():
    """
    Create or load a dataset with text and corresponding stylized summaries
    Format: [{"text": original_text, "summary_formal": formal_summary, "summary_informal": informal_summary, ...}]
    """
    # Configure Gemini API
    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY not found in environment variables or .env file"
        )

    genai.configure(api_key=GEMINI_API_KEY)

    # Initialize the model
    model = genai.GenerativeModel("gemini-2.0-flash")

    # Sample data preparation
    data = []
    for article in load_articles():
        entry = {"text": article}
        for style in ["formal", "informal", "humorous", "poetic"]:
            summary = generate_styled_summary(article, style, model)
            entry[f"summary_{style}"] = summary
        data.append(entry)

    return pd.DataFrame(data)


def prepare_dataset_for_training(df, tokenizer, style="formal"):
    """Convert dataframe to format suitable for training"""

    def preprocess_function(examples):
        # Prepend style token to input
        inputs = [f"[{style.upper()}] {text}" for text in examples["text"]]
        model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

        # Tokenize summaries
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples[f"summary_{style}"],
                max_length=128,
                truncation=True,
                padding="max_length",
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Convert to Hugging Face dataset
    dataset = Dataset.from_pandas(df)

    # Split dataset into train and validation
    dataset = dataset.train_test_split(test_size=0.2)
    dataset = {"train": dataset["train"], "validation": dataset["test"]}

    # Tokenize dataset
    tokenized_dataset = {}
    for split in dataset:
        tokenized_dataset[split] = dataset[split].map(preprocess_function, batched=True)

    return tokenized_dataset
