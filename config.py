from pathlib import Path

def get_config():
    return {
        # training
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 1e-4,

        # model / data
        "seq_len": 128,               # maximum sequence length (source & target)
        "d_model": 512,
        "lang_src": "en",
        "lang_tgt": "it",

        # files / experiment
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,              # e.g. "01" or None to start from scratch
        "tokenizer_file": "tokenizer_{}.json",
        "experiment_name": "runs/tmodel"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = Path(config["model_folder"])
    model_folder.mkdir(parents=True, exist_ok=True)
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(model_folder / model_filename)
