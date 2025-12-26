import streamlit as st
import requests

# Page config
st.set_page_config(page_title="Transformer Translator", page_icon="🌎", layout="centered")

# App title
st.markdown("<h1 style='text-align: center; color: #4B0082;'>Transformer Translator</h1>", unsafe_allow_html=True)
st.write("Translate text between languages using your custom Transformer model.")

# Language dropdown
source_lang = st.selectbox("Source Language", ["English"])
target_lang = st.selectbox("Target Language", ["Spanish"])  # you can add more later

# Text input
text_to_translate = st.text_area("Enter text to translate", height=150)

# Translate button
if st.button("Translate"):
    if text_to_translate.strip() == "":
        st.warning("Please enter text to translate!")
    else:
        with st.spinner("Translating..."):
            response = requests.post(
                "http://localhost:8000/translate",
                json={
                    "source_lang": source_lang.lower(),
                    "target_lang": target_lang.lower(),
                    "text": text_to_translate
                }
            )
            if response.status_code == 200:
                translation = response.json()["translation"]
                st.success("Translation:")
                st.text_area("Translated Text", value=translation, height=150)
            else:
                st.error("Error connecting to backend.")
