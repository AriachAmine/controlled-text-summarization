import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Trainer, TrainingArguments, GenerationConfig


def load_model(model_name="facebook/bart-large-cnn"):
    """
    Load a pre-trained summarization model
    Options: facebook/bart-large-cnn, google/pegasus-xsum, sshleifer/distilbart-cnn-12-6
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer


def fine_tune_model(model, tokenizer, dataset, output_dir="./summarization_model"):
    """Fine-tune model on prepared dataset"""
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        num_train_epochs=50,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
    )

    trainer.train()

    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)

    return model


def generate_stylized_summary(text, model, tokenizer, style="formal", max_length=150):
    """Generate a summary in the specified style"""
    # Prepend style token to input
    styled_input = f"[{style.upper()}] {text}"
    inputs = tokenizer(
        styled_input, return_tensors="pt", max_length=1024, truncation=True
    )

    generation_config = GenerationConfig(
        max_length=max_length,
        min_length=56,
        early_stopping=True,
        num_beams=4,
        length_penalty=2.0,
        no_repeat_ngram_size=3,
        forced_bos_token_id=0,
    )

    summary_ids = model.generate(
        inputs["input_ids"], generation_config=generation_config
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
