from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_summary_model"
TEMPLATES_DIR = BASE_DIR / "templates" if (BASE_DIR / "templates").exists() else BASE_DIR

app = FastAPI(title="Text Summarizer App", description="Text summarization using T5", version="1.0")

# model and tokenizer
model = T5ForConditionalGeneration.from_pretrained(str(MODEL_DIR))
tokenizer = T5Tokenizer.from_pretrained(str(MODEL_DIR))

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# templating
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


class DialogueInput(BaseModel):
    dialogue: str


def clean_data(text):
    text = re.sub(r"<.*?>", " ", text)   # remove tags like <file_photo>
    text = re.sub(r"\s+", " ", text)     # newlines/multiple spaces -> ONE space
    return text.strip().lower()


def summarize_dialogue(dialogue: str) -> str:
    dialogue = "summarize: " + clean_data(dialogue)
    inputs = tokenizer(dialogue, padding="max_length", max_length=512,
                       truncation=True, return_tensors="pt").to(device)
    out = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=60,
        min_length=10,
        num_beams=4,
        no_repeat_ngram_size=3,
        early_stopping=True,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)


@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")