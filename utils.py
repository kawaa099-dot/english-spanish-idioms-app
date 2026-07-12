import json
import sqlite3
import tempfile
from gtts import gTTS
from datasets import load_dataset
from transformers import MarianMTModel, MarianTokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pathlib
import streamlit as st
import random
from functools import lru_cache
from transformers import pipeline
import re

# LOAD IDIOMS
@st.cache_data
def load_idioms(path="idiom2.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned = {}

    for idiom, info in data.items():

        cleaned[idiom.lower()] = {
            "meaning": info.get("meaning", ""),
            "topic": info.get("topic", "General")
        }

    return cleaned

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
    return pipeline(
        "text-generation",
        model="distilgpt2"
    )

generator = load_generator()

#DETECTION IN SENTENCES
# Load ai used in detection of idiom in sentences
@st.cache_resource
def load_detector():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

def detect_idioms_ai(text, idiom_map):
    tokenizer, model = load_detector()
    idiom_list = list(idiom_map.keys())

    prompt = f"""From this list of idioms: {", ".join(idiom_list)}
            Which idiom best matches the meaning of this sentence?
            Sentence: "{text}"
            Answer with only the idiom, exactly as written in the list."""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=20)
    guess = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()

    matches = [i for i in idiom_list if i.lower() == guess]
    if not matches:
        matches = [i for i in idiom_list if i.lower() in guess or guess in i.lower()]

    return matches

def normalize_structure(sentence):
    """
    Simplify sentence structure for repetition detection.
    """

    sentence = sentence.lower()

    # remove punctuation
    sentence = re.sub(r"[^\w\s]", "", sentence)

    # remove common subjects
    starters = [
        "i", "he", "she", "they", "we", "my boss",
        "the company", "someone", "people", "our team"
    ]

    words = sentence.split()

    if len(words) >= 2:
        first_two = " ".join(words[:2])

        if first_two in starters:
            return " ".join(words[2:5])

    return " ".join(words[:3])

def generate_ai_sentence(idiom, examples_map, used_structures):

    subjects = [
        "My boss", "The kids", "A stranger", "Our team",
        "The company", "Her friend", "The teacher",
        "Someone", "People", "The situation"
    ]

    tones = [
        "casual", "professional", "funny",
        "dramatic", "everyday"
    ]

    banned_phrases = [
        "i decided",
        "he decided",
        "she decided",
        "decided to"
    ]

    for _ in range(8):

        try:

            subject = random.choice(subjects)
            tone = random.choice(tones)

            prompt = f"""
            Write one natural {tone} English sentence using the idiom "{idiom}".
            Use "{subject}" as the subject.
            Make it conversational and realistic.
            """

            result = generator(
                prompt,
                max_new_tokens=25,
                do_sample=True,
                temperature=1.0,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2,
                truncation=True
            )

            sentence = result[0]["generated_text"]
            sentence = sentence.replace(prompt, "").strip()

            # Basic validation
            if not sentence:
                continue

            if len(sentence.split()) < 5:
                continue

            if idiom.lower() not in sentence.lower():
                continue

            lower = sentence.lower()

            # Reject repetitive templates
            if any(bad in lower for bad in banned_phrases):
                continue

            # Detect repeated structures
            structure = normalize_structure(sentence)

            if structure in used_structures:
                continue

            used_structures.add(structure)

            # Replace idiom with blank
            sentence = re.sub(
                re.escape(idiom),
                "_____",
                sentence,
                flags=re.IGNORECASE
            )

            return sentence

        except Exception:
            continue

    # ---------- FALLBACK TO DATASET ----------
    examples = examples_map.get(idiom.lower(), [])

    valid_examples = []

    for ex in examples:
        text = ex.get("en", "")

        if idiom.lower() in text.lower():

            lower = text.lower()

            if any(bad in lower for bad in banned_phrases):
                continue

            structure = normalize_structure(text)

            if structure not in used_structures:
                valid_examples.append(text)

    if valid_examples:

        sentence = random.choice(valid_examples)

        used_structures.add(
            normalize_structure(sentence)
        )

        sentence = re.sub(
            re.escape(idiom),
            "_____",
            sentence,
            flags=re.IGNORECASE
        )

        return sentence

    return None

def generate_distractors(correct_idiom, all_idioms):
    pool = [i for i in all_idioms if i != correct_idiom]
    distractors = random.sample(pool, min(3, len(pool)))
    options = distractors + [correct_idiom]
    random.shuffle(options)
    return options

def generate_adaptive_quiz(
    conn,
    idiom_map,
    examples_map,
    used_questions,
    used_structures,
    max_ai_attempts=2
):
    import time
    idioms = list(idiom_map.keys())
    available = [i for i in idioms if i not in used_questions]

    if not available:
        used_questions.clear()
        available = idioms[:]

    weak = get_weak_idioms(conn)
    if weak:
        weak_available = [i for i in available if i in weak]
        if weak_available:
            available = weak_available

    random.shuffle(available)

    # ---------- DATASET FIRST (fast) ----------
    t0 = time.time()
    for idiom in available:
        examples = examples_map.get(idiom.lower(), [])
        for ex in examples:
            sentence = ex.get("en", "")
            if not sentence:
                continue
            pattern = re.compile(r'\b' + re.escape(idiom) + r'\b', re.IGNORECASE)
            if not pattern.search(sentence):
                continue
            structure = normalize_structure(sentence)
            if structure in used_structures:
                continue
            used_structures.add(structure)
            used_questions.add(idiom)
            question_sentence = pattern.sub("_____", sentence)
            wrong_pool = [i for i in idioms if i != idiom]
            wrong_options = random.sample(wrong_pool, min(3, len(wrong_pool)))
            options = wrong_options + [idiom]
            random.shuffle(options)
            print(f"[TIMING] dataset lookup: {time.time() - t0:.2f}s")
            return {
                "question": question_sentence,
                "options": options,
                "answer": idiom
            }
    print(f"[TIMING] dataset lookup (no match): {time.time() - t0:.2f}s")

    # ---------- AI FALLBACK, CAPPED ----------
    t1 = time.time()
    for idiom in available[:max_ai_attempts]:
        sentence = generate_ai_sentence(idiom, examples_map, used_structures)
        if not sentence:
            continue
        pattern = re.compile(r'\b' + re.escape(idiom) + r'\b', re.IGNORECASE)
        if not pattern.search(sentence):
            continue
        used_questions.add(idiom)
        question_sentence = pattern.sub("_____", sentence)
        wrong_pool = [i for i in idioms if i != idiom]
        wrong_options = random.sample(wrong_pool, min(3, len(wrong_pool)))
        options = wrong_options + [idiom]
        random.shuffle(options)
        print(f"[TIMING] AI generation: {time.time() - t1:.2f}s")
        return {
            "question": question_sentence,
            "options": options,
            "answer": idiom
        }
    print(f"[TIMING] AI generation (failed): {time.time() - t1:.2f}s")

    used_questions.clear()
    used_structures.clear()
    return None
    
# DETECT IDIOMS
def detect_idioms(text, idioms):
    text_lower = text.lower()
    return [i for i in idioms if i.lower() in text_lower]
    
