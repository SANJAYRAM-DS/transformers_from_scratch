import streamlit as st
from transformers import MarianTokenizer, MarianMTModel
import torch

# Page config
st.set_page_config(page_title="Transformer Translator 🌎", page_icon="🌎", layout="centered")

st.title("🌎 Transformer Translator")
st.write("English → Spanish Translation")

# Load model once and cache it
@st.cache_resource
def load_model():
    tokenizer = MarianTokenizer.from_pretrained("Sanjayramdata/Translatorr")
    model = MarianMTModel.from_pretrained("Sanjayramdata/Translatorr")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# Text input
text = st.text_area("Enter English text", height=150)

# Translate button
if st.button("Translate"):
    if text.strip() == "":
        st.warning("Please enter text")
    else:
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors="pt")
            outputs = model.generate(**inputs, max_length=128)
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

        st.success("Translation:")
        st.text_area("Spanish", translation, height=150)
