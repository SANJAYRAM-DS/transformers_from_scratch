# src/inference.py
"""
Load the trained model and run translations.
Usage:
    python src/inference.py --text "Hello world"
"""
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from src.configs import get_config

cfg = get_config()

def build_translator(model_dir=None):
    model_dir = model_dir or cfg["output_dir"]
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    device = 0 if (torch.cuda.is_available()) else -1
    translator = pipeline("translation", model=model, tokenizer=tokenizer, device=device)
    return translator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=False, help="Text to translate")
    parser.add_argument("--model_dir", type=str, required=False, help="Model directory (overrides config)")
    args = parser.parse_args()

    translator = build_translator(args.model_dir)

    if args.text:
        out = translator(args.text, max_length=cfg["max_target_length"])
        print("SOURCE:", args.text)
        print("TRANSLATION:", out[0].get("translation_text") if isinstance(out, list) else out)
    else:
        print("Interactive translation (type 'exit' to quit).")
        while True:
            txt = input("SOURCE> ").strip()
            if txt.lower() in ("exit", "quit"):
                break
            out = translator(txt, max_length=cfg["max_target_length"])
            print("TRANSLATION:", out[0].get("translation_text"))
