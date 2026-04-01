import json
import sqlite3
import tempfile
from gtts import gTTS
from datasets import load_dataset
from transformers import MarianMTModel, MarianTokenizer
import pathlib
import streamlit as st
import random
from functools import lru_cache
from transformers import pipeline

# LOAD IDIOMS
@st.cache_data
def load_idioms(path="idioms.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k.lower(): v for k, v in data.items()}

# AUDIO
CACHE_DIR = pathlib.Path("audio_cache")
CACHE_DIR.mkdir(exist_ok=True)

@st.cache_data
def generate_audio(text, lang="en"):
    safe_text = text.replace(" ", "_")
    file_path = CACHE_DIR / f"{safe_text}.mp3"

    if not file_path.exists():
        tts = gTTS(text=text, lang=lang)
        tts.save(file_path)

    return str(file_path)

# SQLITE DATABASE
def init_db():
    conn = sqlite3.connect("idioms.db", check_same_thread=False)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS favorite (
            idiom TEXT PRIMARY KEY,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS analytics (
            idiom TEXT PRIMARY KEY,
            attempts INTEGER DEFAULT 0,
            correct INTEGER DEFAULT 0
        )
    """)

    conn.commit()
    return conn

def add_favorite(conn, idiom):
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO favorite (idiom) VALUES (?)", (idiom,))
    conn.commit()

def get_favorite(conn):
    c = conn.cursor()
    rows = c.execute("SELECT idiom FROM favorite ORDER BY timestamp DESC").fetchall()
    return [r[0] for r in rows]

def remove_favorite(conn, idiom):
    c = conn.cursor()
    c.execute("DELETE FROM favorite WHERE idiom = ?", (idiom,))
    conn.commit()

# ANALYTICS
def update_analytics(conn, idiom, is_correct):
    c = conn.cursor()

    c.execute("""
        INSERT INTO analytics (idiom, attempts, correct)
        VALUES (?, 0, 0)
        ON CONFLICT(idiom) DO NOTHING
    """, (idiom,))

    c.execute("""
        UPDATE analytics
        SET attempts = attempts + 1,
            correct = correct + ?
        WHERE idiom = ?
    """, (1 if is_correct else 0, idiom))

    conn.commit()

def get_learning_stats(conn):
    c = conn.cursor()
    rows = c.execute("SELECT idiom, attempts, correct FROM analytics").fetchall()

    stats = []
    for idiom, attempts, correct in rows:
        acc = correct / attempts if attempts else 0
        stats.append((idiom, attempts, correct, acc))

    return stats

def get_weak_idioms(conn):
    c = conn.cursor()
    rows = c.execute("""
        SELECT idiom, attempts, correct
        FROM analytics
        WHERE attempts >= 2
    """).fetchall()

    weak = []
    for idiom, attempts, correct in rows:
        accuracy = correct / attempts if attempts else 0
        if accuracy < 0.6:
            weak.append(idiom)

    return weak

# TRANSLATION MODEL
@st.cache_resource
def load_translation_model():
    model_name = "Helsinki-NLP/opus-mt-en-es"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

def translate_literal(text):
    tokenizer, model = load_translation_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    translated = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# EXAMPLES DATASETS
@st.cache_data(show_spinner=True)
def build_examples_map():
    examples_map = {}

    def add_example(key, en, es):
        key = key.lower().strip()
        if key not in examples_map:
            examples_map[key] = []
        if en or es:
            examples_map[key].append({"en": en or "", "es": es or ""})

    try:
        ds1 = load_dataset("fdelucaf/IdioTS")
        for row in ds1["train"]:
            if row["sentence_has_idiom"]:
                add_example(row["idiom"], row.get("en", ""), row.get("es", ""))

        ds2 = load_dataset("UCSC-Admire/idiom-SFT-dataset-561-2024-12-06_00-40-30")
        for row in ds2["train"]:
            idiom = row.get("idiom") or row.get("Idiomatic Expression") or ""
            en = row.get("en") or row.get("English") or ""
            es = row.get("es") or row.get("Spanish") or ""
            if idiom:
                add_example(idiom, en, es)

    except Exception:
        pass

    return examples_map

# QUIZ GENERATION

# Load once
@st.cache_resource
def load_generator():
    return pipeline("text-generation", model="google/flan-t5-base")

generator = load_generator()

def generate_ai_sentence(idiom, examples_map):
    prompt = (
        f"Write one natural English sentence using the idiom '{idiom}'. "
        f"Replace the idiom with a blank (_____). Only output the sentence."
    )

    try:
        result = generator(
            prompt,
            max_new_tokens=40,
            do_sample=True,
            temperature=0.9
        )
        text = result[0]['generated_text'].strip()

        # Remove prompt if echoed
        if prompt.lower() in text.lower():
            text = text.replace(prompt, "").strip()

        # Check if valid
        if "_____" in text and len(text) > 20:
            return text

    except:
        pass

    #FALLBACK → dataset example
    examples = examples_map.get(idiom.lower(), [])
    if examples:
        sentence = random.choice(examples)["en"]
        return sentence.replace(idiom, "_____")

    

def generate_distractors(correct_idiom, all_idioms):
    pool = [i for i in all_idioms if i != correct_idiom]
    distractors = random.sample(pool, min(3, len(pool)))
    options = distractors + [correct_idiom]
    random.shuffle(options)
    return options

def generate_ai_question_dynamic(all_idioms, examples_map):
    correct = random.choice(all_idioms)

    sentence = generate_ai_sentence(correct, examples_map)
    options = generate_distractors(correct, all_idioms)

    return {
        "question": sentence,
        "options": options,
        "answer": correct
    }

# DETECT IDIOMS
def detect_idioms(text, idioms):
    text_lower = text.lower()
    return [i for i in idioms if i.lower() in text_lower]