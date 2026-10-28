# Text Summarizer Web App

A dialogue and text summarization web application powered by **FastAPI** and a fine-tuned **T5 Transformer** model.

## Features
- **FastAPI Backend**: Fast, lightweight API with interactive UI and endpoint documentation (`/docs`).
- **T5 Transformer**: Pretrained/fine-tuned conditional generation model tailored for summarization.
- **Interactive Web Interface**: Clean HTML/CSS interface served via Jinja2 templates.
- **Git LFS Enabled**: Pre-configured Git Large File Storage for tracking transformer model weights (`*.safetensors`).

## Project Structure
```
.
├── app.py                     # FastAPI application & summarization pipeline
├── saved_summary_model/       # T5 model weights and tokenizer configurations
│   ├── config.json
│   ├── generation_config.json
│   ├── model.safetensors      # Model weights (tracked with Git LFS)
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── templates/
│   └── index.html             # UI frontend
├── requirements.txt           # Python dependencies
├── .gitattributes             # Git LFS attributes
└── README.md
```

## Getting Started

### 1. Clone the Repository
```bash
git clone <YOUR_REPOSITORY_URL>
cd <REPOSITORY_FOLDER>
git lfs pull
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
uvicorn app:app --reload
```
Open your browser and navigate to `http://localhost:8000`.
