from fastapi import FastAPI
from pydantic import BaseModel
from transformers import MarianTokenizer, MarianMTModel
from fastapi.middleware.cors import CORSMiddleware
import torch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME = "Sanjayramdata/Translatorr"

tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model = MarianMTModel.from_pretrained(MODEL_NAME)
model.eval()

class TranslationRequest(BaseModel):
    source_lang: str
    target_lang: str
    text: str

@app.post("/translate")
def translate(req: TranslationRequest):
    inputs = tokenizer(req.text, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=128,
            num_beams=4
        )

    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"translation": translation}
