import os
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
import evaluate
from src.configs import get_config

cfg = get_config()


def postprocess_text(preds, labels):
    preds = [p.strip() for p in preds]
    labels = [l.strip() for l in labels]
    return preds, labels


def compute_metrics(eval_pred):
    metric = evaluate.load("sacrebleu")
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    decoded_preds = tokenizer.batch_decode(
        np.argmax(logits, axis=-1), skip_special_tokens=True
    ) if logits.ndim == 3 else tokenizer.batch_decode(logits, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    return metric.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])


if __name__ == "__main__":
    # Load dataset
    dataset = load_dataset("opus_books", "en-es")
    print(f"Loaded HF dataset opus_books / en-es with {len(dataset['train'])} rows")

    # Split into train/test
    dataset = dataset["train"].train_test_split(test_size=0.1)
    print(dataset)

    # Load tokenizer and model
    model_name = cfg["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Loading model:", model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Tokenize
    def preprocess_function(examples):
        inputs = [ex["en"] for ex in examples["translation"]]
        targets = [ex["es"] for ex in examples["translation"]]
        model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
        labels = tokenizer(text_target=targets, max_length=128, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_datasets = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="longest")

    # Training args
    try:
        training_args = Seq2SeqTrainingArguments(
            output_dir=cfg["output_dir"],
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=cfg["logging_steps"],
            per_device_train_batch_size=cfg["per_device_train_batch_size"],
            per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            num_train_epochs=cfg["num_train_epochs"],
            learning_rate=cfg["learning_rate"],
            predict_with_generate=True,
            fp16=False,
            save_total_limit=cfg["save_total_limit"],
            seed=cfg["seed"],
            load_best_model_at_end=False,
            remove_unused_columns=False,
            report_to="none",
        )
    except TypeError:
        print("Using fallback: `eval_strategy` instead of `evaluation_strategy`")
        training_args = Seq2SeqTrainingArguments(
            output_dir=cfg["output_dir"],
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=cfg["logging_steps"],
            per_device_train_batch_size=cfg["per_device_train_batch_size"],
            per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            num_train_epochs=cfg["num_train_epochs"],
            learning_rate=cfg["learning_rate"],
            predict_with_generate=True,
            fp16=False,
            save_total_limit=cfg["save_total_limit"],
            seed=cfg["seed"],
            load_best_model_at_end=False,
            remove_unused_columns=False,
            report_to="none",
        )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=None,
    )

    print("Starting training. This may be slow on CPU — GPU recommended.")
    trainer.train()

    os.makedirs(cfg["output_dir"], exist_ok=True)
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print("Model and tokenizer saved to:", cfg["output_dir"])
