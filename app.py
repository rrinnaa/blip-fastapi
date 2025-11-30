from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    M2M100Tokenizer,
    M2M100ForConditionalGeneration
)
import io

app = FastAPI()

def get_device():
    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


device = get_device()

processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl"
).to(device)

translator_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
translator_model = M2M100ForConditionalGeneration.from_pretrained(
    "facebook/m2m100_418M"
).to(device)

PROMPT = (
    "Describe exactly what is visible in the image. "
    "Do not guess location, emotions, actions, or background. "
    "Keep it short, 3-6 words, only visible objects."
)

def shorten_text(text: str, max_words: int = 6) -> str:
    words = text.strip().split()
    return " ".join(words[:max_words]) if words else "no clear objects"

def translate_to_ru(text: str) -> str:
    translator_tokenizer.src_lang = "en"
    encoded = translator_tokenizer(text, return_tensors="pt").to(device)
    generated = translator_model.generate(
        **encoded,
        forced_bos_token_id=translator_tokenizer.get_lang_id("ru"),
        max_new_tokens=50
    )
    return translator_tokenizer.decode(generated[0], skip_special_tokens=True)

@app.post("/describe")
async def describe_image(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")

    inputs = processor(images=image, text=PROMPT, return_tensors="pt").to(device)

    caption_en = ""
    for _ in range(2):
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=100,
                num_beams=8,
                no_repeat_ngram_size=2,
                do_sample=False,
                early_stopping=True
            )
        caption_en = processor.decode(output[0], skip_special_tokens=True).strip()
        if caption_en:
            break

    caption_en_short = shorten_text(caption_en)
    caption_ru = translate_to_ru(caption_en_short)

    return {
        "caption_en_raw": caption_en,
        "caption_en": caption_en_short,
        "caption_ru": caption_ru
    }