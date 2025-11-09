# Transformer-based Neural Machine Translation (NMT)

A **Transformer model** trained using **Hugging Face ** for **English ↔ Regional Language** translation.  
This project demonstrates how to train, fine-tune, and deploy a lightweight translation model locally.

---

## Overview

This project implements a **seq2seq Transformer architecture** for real-time text translation.  
It is based on the original **"Attention Is All You Need"** paper by Vaswani et al. (2017).

Key highlights:
- Built with **Hugging Face Transformers**
- Supports **custom domain-specific training** (Agriculture dataset)
- Includes **tokenization, dataset preprocessing, and training pipeline**
- Easily deployable for **RAG/LLM integration** or **API translation services**

---

## Architecture

```

Input Text → Tokenizer → Encoder (Transformer) → Decoder (Transformer) → Output Translation

```

- **Encoder**: Processes input sentence and generates contextual embeddings  
- **Decoder**: Generates translated text token-by-token using attention  
- **Attention Mechanism**: Learns relationships between source and target words  
- **Embedding Layer**: Converts words into dense vector representations  

---

## Project Structure

```

translator-transformer/
├── src/
│   ├── dataset_preparation.py
│   ├── model_training.py
│   ├── configs.py
│   └── inference.py
├── requirements.txt
├── README.md
└── app.py

````

---

## Setup Instructions

### 1️.Clone the repository
```bash
git clone https://github.com/<your-username>/translator-transformer.git
cd translator-transformer
````

### 2️.Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate    # Windows
# or
source venv/bin/activate # Linux/Mac
```

### 3️.Install dependencies

```bash
pip install -r requirements.txt
```

### 4️.Train the model

```bash
python src/model_training.py
```

### 5️.Run inference

```bash
python src/inference.py --text "Translate this to Hindi"
```

---

## Example Usage

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("models/transformer_translation_model")
model = AutoModelForSeq2SeqLM.from_pretrained("models/transformer_translation_model")

text = "Crops need sufficient sunlight and water."
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## Training Configuration

| Parameter     | Value                               |
| ------------- | ----------------------------------- |
| Model Type    | Transformer (Encoder-Decoder)       |
| Tokenizer     | SentencePiece / BPE                 |
| Optimizer     | AdamW                               |
| Learning Rate | 5e-5                                |
| Batch Size    | 16                                  |
| Framework     | Hugging Face Transformers           |
| Hardware      | Local CPU (Ryzen 5 5500U, 16GB RAM) |

---

## Future Enhancements

* Integrate with **Retrieval-Augmented Generation (RAG)** for domain-specific question answering
* Fine-tune with **agriculture knowledge base**
* Deploy using **FastAPI** for real-time translation API

---

## Dependencies

```
transformers
datasets
torch
pandas
numpy
tqdm
sentencepiece
```

---

## License

This project is released under the **MIT License**.

---

## Author

**SanjayRam M**
- [GitHub](https://github.com/SANJAYRAM-DS>) | ✉️ [sanjayram.data@gmail.com](mailto:sanjayram.data@gmail.com)
- *"Building AI solutions for language understanding and domain translation."*
