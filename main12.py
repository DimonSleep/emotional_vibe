import os
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import traceback
import logging
import random

import torch
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import PyPDF2
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# --- CONFIG & ENV ---
print("=" * 60)
print("AI SUPORT EMOȚIONAL - ASISTENT EMPATIC CU BAZĂ ACADEMICĂ")
print("=" * 60)
print(f"Versiune PyTorch: {torch.__version__}")
print(f"CUDA disponibil: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU detectat: {torch.cuda.get_device_name(0)}")
    print(f"Memorie GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    DEVICE = "cuda"
    torch.cuda.empty_cache()
else:
    print("GPU nu este detectat! Se folosește CPU")
    DEVICE = "cpu"

logging.basicConfig(level=logging.INFO)
load_dotenv()

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
MODEL_NAME = "microsoft/DialoGPT-medium"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DATA_DIR = Path("data")
TEMPLATES_DIR = Path("templates")
VECTOR_STORE_PATH = DATA_DIR / "vector_store.index"
METADATA_PATH = DATA_DIR / "metadata.json"
TRAINING_DATA_PATH = DATA_DIR / "training_data.json"
DATA_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)

training_data = []
chat_history = []

embedding_model = None
index = None
metadata = []

# --- RAG ---
def init_rag():
    global embedding_model, index, metadata, training_data
    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        if METADATA_PATH.exists():
            with open(METADATA_PATH, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        if TRAINING_DATA_PATH.exists():
            with open(TRAINING_DATA_PATH, 'r', encoding='utf-8') as f:
                training_data = json.load(f)
        if VECTOR_STORE_PATH.exists():
            index = faiss.read_index(str(VECTOR_STORE_PATH))
        else:
            dim = embedding_model.get_sentence_embedding_dimension()
            index = faiss.IndexFlatL2(dim)
    except Exception as e:
        print(f"Eroare la inițializarea RAG: {e}")
        traceback.print_exc()

def save_rag():
    global index, metadata, training_data
    try:
        faiss.write_index(index, str(VECTOR_STORE_PATH))
        with open(METADATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        with open(TRAINING_DATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Eroare la salvarea RAG: {e}")

def add_to_rag(texts: List[str], source: str):
    global metadata, index, embedding_model
    try:
        embeddings = embedding_model.encode(texts)
        start_id = len(metadata)
        for i, text in enumerate(texts):
            metadata.append({
                "text": text,
                "source": source,
                "id": start_id + i,
                "timestamp": datetime.now().isoformat()
            })
        index.add(np.array(embeddings, dtype=np.float32))
    except Exception as e:
        print(f"Eroare la adăugarea în RAG: {e}")

def search_rag(query: str, k: int = 3) -> List[Dict]:
    try:
        if not index or index.ntotal == 0:
            return []
        query_vec = embedding_model.encode([query])
        distances, indices = index.search(np.array(query_vec, dtype=np.float32), k)
        results = []
        for idx in indices[0]:
            if idx < len(metadata):
                results.append(metadata[idx])
        return results
    except Exception as e:
        print(f"Eroare la căutarea în RAG: {e}")
        return []

def get_academic_stats():
    academic_count = 0
    total_fragments = index.ntotal if index else 0
    for item in meta
        if any(keyword in item['source'].lower() for keyword in ['teza', 'teză', 'doctorat', 'academic']):
            academic_count += 1
    return {
        "total_fragments": total_fragments,
        "academic_fragments": academic_count,
        "academic_percentage": (academic_count / total_fragments * 100) if total_fragments > 0 else 0
    }

def add_academic_demo_data():
    academic_demo = [
        "Studiile arată că episoadele de manie pot fi gestionate prin rutine de somn regulate și evitarea stimulentilor",
        "Cercetările indică că tehnicile de grounding pot reduce intensitatea episoadelor de panică cu până la 70%",
        "Terapia dialectică comportamentală (DBT) s-a dovedit eficientă în gestionarea emoțiilor intense și a impulsurilor",
    ]
    add_to_rag(academic_demo, "suport_emotional_academic_specific")
    demo_dialogues = [
        {
            "user": "Am episoade de manie și mă simt copleșit",
            "assistant": "Înțeleg că episoadele de manie pot simți ca un tsunami emoțional. E epuizant să simți că nu ai control. Rutinele regulate de somn și evitarea stimulentilor pot ajuta mult.",
            "source": "demo_empatic_manie"
        },
        {
            "user": "Mă simt în stare de neputință totală",
            "assistant": "Simțul de neputință poate fi covârșitor. Când totul pare imposibil, chiar și respirația pare grea. Dar faptul că vorbești despre asta arată o forță incredibilă.",
            "source": "demo_empatic_neputinta"
        }
    ]
    training_data.extend(demo_dialogues)
    save_rag()

class RealAI:
    def __init__(self, model_name=MODEL_NAME):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.loaded = False
        self.device = DEVICE
        print(f"AI-ul va rula pe: {self.device.upper()}")

    def load(self):
        if self.loaded:
            return
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.model = self.model.to(self.device)
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                max_new_tokens=120,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
            self.loaded = True
        except Exception as e:
            traceback.print_exc()
            self.loaded = False

    def respond(self, message: str, history: List[Dict] = None) -> str:
        if not self.loaded:
            return "Sistemul se încarcă, te rog să aștepți..."
        try:
            relevant_docs = search_rag(message)
            emotional_support_responses = {
                "singur": ["Te simți singur. Este greu. Nu ești singur în acest sentiment."],
                "depresiv": ["Depresia poate fi ca un nor greu, dar ai curaj și putere."],
                "anxios": ["Anxietatea ne face să simțim că pierdem controlul. E normal - ce te-ar ajuta să te calmezi?"],
                "stresat": ["Stresul e copleșitor, din păcate. E important să iei pauze."],
            }
            message_lower = message.lower()
            for keyword, responses in emotional_support_responses.items():
                if keyword in message_lower:
                    response = random.choice(responses)
                    if relevant_docs:
                        academic_info = relevant_docs[0]['text'][:100] + "..." if len(relevant_docs[0]['text']) > 100 else relevant_docs[0]['text']
                        response += f"\n\nStudii arată că: {academic_info}"
                    return response
            general_support = [
                "Înțeleg că treci prin ceva dificil. Te ascult.",
                "Mulțumesc că împărtășești asta cu mine.",
                "Te aud. Uneori doar vorbitul contează.",
                "Sunt aici pentru tine."
            ]
            response = random.choice(general_support)
            if relevant_docs:
                academic_info = relevant_docs[0]['text'][:120] + "..." if len(relevant_docs[0]['text']) > 120 else relevant_docs[0]['text']
                response += f"\n\nDe știut: {academic_info}"
            return response
        except Exception as e:
            traceback.print_exc()
            return "Îmi pare rău, am o problemă tehnică. Te rog să încerci din nou."

# --- Setup AI and RAG ---
init_rag()
add_academic_demo_data()
ai = RealAI()
ai.load()

# --- FLASK WEB SERVER ---
app = Flask(__name__, template_folder="templates")
CORS(app)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat_api():
    msg = request.json.get("message", "")
    resp = ai.respond(msg)
    return jsonify({"response": resp})

# --- TELEGRAM BOT ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "AI Suport Emoțional - Asistent Empatic cu bază academică. Trimite un mesaj sau atașează PDF de studiu."
    )

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    stats = get_academic_stats()
    pdf_count = len(list(DATA_DIR.glob("*.pdf")))
    await update.message.reply_text(
        f"PDF-uri procesate: {pdf_count}\n"
        f"Dialoguri: {len(training_data)}\n"
        f"Fragmente RAG: {stats['total_fragments']}\n"
        f"Conținut academic: {stats['academic_fragments']} fragmente\n"
        f"Status AI: {'Activ' if ai.loaded else 'Se încarcă...'}"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    ai_response = ai.respond(user_message)
    await update.message.reply_text(ai_response)

async def handle_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document
    if not doc.file_name.lower().endswith('.pdf'):
        await update.message.reply_text("Doar fișiere PDF!")
        return
    try:
        await update.message.reply_text("Procesez PDF-ul pentru antrenament...")
        file = await context.bot.get_file(doc.file_id)
        pdf_path = DATA_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{doc.file_name}"
        await file.download_to_drive(str(pdf_path))

        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = "".join(page.extract_text() or "" for page in reader.pages)
            lines = [l.strip() for l in text.split('\n') if l.strip()]

        dialogues = []
        for i in range(0, len(lines) - 1, 2):
            if i+1 < len(lines):
                user_line = lines[i]
                assistant_line = lines[i+1]
                if (len(user_line) > 15 and len(assistant_line) > 15):
                    dialogues.append({
                        "user": user_line[:300],
                        "assistant": assistant_line[:300],
                        "timestamp": datetime.now().isoformat(),
                        "source": doc.file_name
                    })
        training_data.extend(dialogues)

        chunks = []
        current_chunk = ""
        for line in lines:
            if len(current_chunk) + len(line) > 300:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = line
            else:
                current_chunk += " " + line
        if current_chunk:
            chunks.append(current_chunk)
        add_to_rag(chunks, doc.file_name)
        save_rag()
        await update.message.reply_text(
            f"PDF procesat!\nDialoguri extrase: {len(dialogues)}\nFragmente adăugate: {len(chunks)}\nTotal dialoguri: {len(training_data)}\nTotal fragmente: {index.ntotal if index else 0}"
        )
    except Exception as e:
        await update.message.reply_text(f"Eroare: {str(e)[:120]}...")

def run_telegram():
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.Document.PDF, handle_pdf))
    application.run_polling()

if __name__ == "__main__":
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=8080), daemon=True).start()
    run_telegram()
