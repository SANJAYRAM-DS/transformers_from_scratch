# app.py
"""
Minimal FastAPI app for translation inference (loads the trained model on startup).
Run:
    uvicorn app:app --reload
"""
from fastapi import FastAPI
from pydantic import BaseModel
from src.inference import build_translator
from src.configs import get_config

cfg = get_config()

app = FastAPI(title="Translator API")

translator = None

class TranslateRequest(BaseModel):
    text: str
    model_dir: str = None
    max_length: int = cfg["max_target_length"]

@app.on_event("startup")
def load_model():
    global translator
    translator = build_translator(cfg["output_dir"])

@app.post("/translate")
def translate(req: TranslateRequest):
    global translator
    if req.model_dir:
        # reload translator if different model requested
        translator = build_translator(req.model_dir)
    out = translator(req.text, max_length=req.max_length)
    return {"source": req.text, "translation": out[0].get("translation_text")}
