import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import time
import config

tokenizer = None
model = None

def initialize_model():
    global tokenizer, model
    print("Initializing translation model...")
    start = time.time()
    _tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, src_lang="eng_Latn")
    _model = AutoModelForSeq2SeqLM.from_pretrained(
        config.MODEL_NAME,
        torch_dtype=torch.float16 if config.DEVICE == "cuda" else torch.float32
    ).to(config.DEVICE).eval()
    print(f"Model loaded in {time.time()-start:.2f}s")
    tokenizer = _tokenizer
    model = _model
    return _tokenizer, _model

def get_model_tokenizer():
    return tokenizer, model

def set_model_tokenizer(new_tokenizer, new_model):
    global tokenizer, model
    tokenizer = new_tokenizer
    model = new_model

initialize_model()