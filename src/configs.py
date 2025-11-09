# src/configs.py
from pathlib import Path

def get_config():
    """
    Central config. Edit this to change model / dataset choices.
    """
    base = Path('.')
    return {
        "model_name": "Helsinki-NLP/opus-mt-en-es",
        "output_dir": str(base / "models" / "transformer_translation_model"),
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "gradient_accumulation_steps": 8,
        "num_train_epochs": 3,
        "learning_rate": 5e-5,
        "max_source_length": 128,
        "max_target_length": 128,
        "save_total_limit": 3,
        "logging_steps": 200,
        "seed": 42,
        # dataset
        "dataset_name": "opus_books",
        "dataset_config_name": "en-es",  # will be tried; if not available the script will fall back to loading provided CSVs
        # local files fallback (use if you have train/val CSVs)
        "local_train_csv": "data/train.csv",
        "local_val_csv": "data/val.csv",
    }
