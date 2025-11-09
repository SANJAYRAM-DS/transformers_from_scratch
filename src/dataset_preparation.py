from datasets import load_dataset, Dataset, DatasetDict
import os
import pandas as pd
from transformers import AutoTokenizer
from src.configs import get_config

cfg = get_config()

def load_hf_dataset():
    try:
        ds = load_dataset(cfg["dataset_name"], cfg["dataset_config_name"], split="train")
        print(f"Loaded HF dataset {cfg['dataset_name']} / {cfg['dataset_config_name']} with {len(ds)} rows")
        # hf dataset uses column 'translation' like {'en':..., 'ta':...}
        # convert into simple columns 'source' and 'target'
        def convert_translation(example):
            t = example["translation"]
            src = t.get("en") or t.get("src") or t.get("english")
            tgt = t.get("ta") or t.get("tamil") or t.get(cfg["dataset_config_name"].split('-')[-1])
            return {"source": src, "target": tgt}
        ds = ds.map(lambda ex: convert_translation(ex), remove_columns=ds.column_names)
        return ds.train_test_split(test_size=0.05, seed=cfg["seed"])
    except Exception as e:
        print("Could not load HF dataset (or config not available):", e)
        return None

def load_local_csvs():
    train_csv = cfg["local_train_csv"]
    val_csv = cfg["local_val_csv"]
    if os.path.exists(train_csv) and os.path.exists(val_csv):
        df_train = pd.read_csv(train_csv)
        df_val = pd.read_csv(val_csv)
        # try to detect columns
        for cands in (("en", "ta"), ("source", "target"), ("src", "tgt")):
            if set(cands).issubset(df_train.columns):
                df_train = df_train.rename(columns={cands[0]: "source", cands[1]: "target"})
                df_val = df_val.rename(columns={cands[0]: "source", cands[1]: "target"})
                break
        # drop na
        df_train = df_train.dropna(subset=["source", "target"])
        df_val = df_val.dropna(subset=["source", "target"])
        ds_train = Dataset.from_pandas(df_train.reset_index(drop=True))
        ds_val = Dataset.from_pandas(df_val.reset_index(drop=True))
        return DatasetDict({"train": ds_train, "validation": ds_val})
    else:
        print("Local CSVs not found at", train_csv, val_csv)
        return None

def tokenize_datasets(tokenizer, datasets, max_source_len, max_target_len):
    def preprocess_function(examples):
        # support dict-of-lists input
        inputs = examples["source"]
        targets = examples["target"]
        model_inputs = tokenizer(inputs, max_length=max_source_len, truncation=True, padding="max_length")
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_len, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    tokenized = datasets.map(preprocess_function, batched=True, remove_columns=datasets["train"].column_names)
    return tokenized

def prepare_data(tokenizer_name=None):
    cfg = get_config()
    ds_dict = load_hf_dataset()
    if ds_dict is None:
        ds_dict = load_local_csvs()
    if ds_dict is None:
        raise RuntimeError("No dataset available. Provide HF 'opus_books' config or place CSVs in data/")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or cfg["model_name"], use_fast=True)
    tokenized = tokenize_datasets(tokenizer, ds_dict, cfg["max_source_length"], cfg["max_target_length"])
    return tokenized, tokenizer

if __name__ == "__main__":
    tokenized, tokenizer = prepare_data()
    print("Tokenized dataset keys:", tokenized.keys())
    print(tokenized["train"][0])
