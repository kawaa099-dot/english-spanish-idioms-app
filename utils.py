import json
import sqlite3
import tempfile
from gtts import gTTS
from datasets import load_dataset
from transformers import MarianMTModel, MarianTokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
import pathlib
import streamlit as st
import random
from functools import lru_cache
from transformers import pipeline
import re

def normalize_idiom(text):
    """
    Normalize an idiom string for consistent dictionary keys / matching:
    - lowercase
    - strip leading/trailing whitespace
    - strip a leading "to " (dataset idioms sometimes include the infinitive marker)
    """
    text = text.strip().lower()
    if text.startswith("to "):
        text = text[3:].strip()
    return text
    
# LOAD IDIOMS
@st.cache_data
def load_idioms(path="idiom2.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned = {}

    for idiom, info in data.items():

        cleaned[normalize_idiom(idiom)] = {
            "meaning": info.get("meaning", ""),
            "topic": info.get("topic", "General"),
            "gloss": info.get("gloss", "")   # ← add this line
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
        #key = key.lower().strip()
        key = normalize_idiom(key)
        if key not in examples_map:
            examples_map[key] = []
        if en or es:
            examples_map[key].append({"en": en or "", "es": es or ""})

    try:
        #ds1 = load_dataset("fdelucaf/IdioTS")
        #for row in ds1["train"]:
        #    if row["sentence_has_idiom"]:
        #        add_example(row["idiom"], row.get("en", ""), row.get("es", ""))

        ds1 = load_dataset("fdelucaf/IdioTS")
        for row in ds1["train"]:
            value = str(row["sentence_has_idiom"]).strip().lower()
            if value == "true":
                add_example(row["idiom"], row.get("en", ""), row.get("es", ""))

        ds2 = load_dataset("UCSC-Admire/idiom-SFT-dataset-561-2024-12-06_00-40-30")
        #for row in ds2["train"]:
        #    idiom = row.get("idiom") or row.get("Idiomatic Expression") or ""
        #    en = row.get("en") or row.get("English") or ""
        #    es = row.get("es") or row.get("Spanish") or ""
        #    if idiom:
        #        add_example(idiom, en, es)
        for row in ds2["train"]:
            usage_type = (row.get("type") or row.get("label") or "").lower()
            if usage_type != "idiomatic":
                continue  # skip literal / distractor sentences

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
#def load_generator():
#    return pipeline(
#        "text-generation",
#        model="distilgpt2"
#    )
    
#generator = load_generator()
@st.cache_resource
def load_generator():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


#DETECTION IN SENTENCES
 #   return SentenceTransformer('all-MiniLM-L6-v2')
@st.cache_resource
def load_similarity_model():
    return SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

def build_embedding_text(idiom, info):
    gloss = info.get("gloss", "")
    return f"{gloss}. {info['meaning']}" if gloss else info["meaning"]
    
@st.cache_data(show_spinner=False)
def _get_idiom_meaning_embeddings(idiom_map_tuple):
    """
    Cached separately from the model itself, so meanings only get
    re-embedded when idiom_map actually changes, not on every call.
    idiom_map_tuple: tuple of (idiom, meaning) pairs, since dicts aren't hashable
    for st.cache_data.
    """
    model = load_similarity_model()
    idioms = [pair[0] for pair in idiom_map_tuple]
    meanings = [pair[1] for pair in idiom_map_tuple]
    embeddings = model.encode(meanings, convert_to_tensor=True)
    return idioms, embeddings
# DETECT IDIOMS
def detect_idioms_ai(text, idiom_map, threshold=0.5):
    model = load_similarity_model()

    idiom_map_tuple = tuple(
        (idiom, build_embedding_text(idiom, info))  
        #(idiom, info["meaning"]) 
        for idiom, info in idiom_map.items()
    )
    idioms, meaning_embeddings = _get_idiom_meaning_embeddings(idiom_map_tuple)

    text_embedding = model.encode(text, convert_to_tensor=True)
    scores = util.cos_sim(text_embedding, meaning_embeddings)[0]

    best_idx = int(scores.argmax())
    best_score = float(scores[best_idx])

    if best_score < threshold:
        return []

    return [idioms[best_idx]]

def detect_idioms(text, idioms):
    text_lower = text.lower()
    return [i for i in idioms if i.lower() in text_lower]

#def detect_idioms_ai(text, idiom_map, top_k=3, threshold=0.35):
 #   model = load_similarity_model()

  #  idiom_map_tuple = tuple(
   #     (idiom, info["meaning"]) for idiom, info in idiom_map.items()
    #)
  #  idioms, meaning_embeddings = _get_idiom_meaning_embeddings(idiom_map_tuple)

   # text_embedding = model.encode(text, convert_to_tensor=True)
    #scores = util.cos_sim(text_embedding, meaning_embeddings)[0]

   # ranked = sorted(
   #     zip(idioms, scores.tolist()),
   #     key=lambda pair: pair[1],
  #      reverse=True
 #   )

#    suggestions = [idiom for idiom, score in ranked[:top_k] if score >= threshold]
#    return suggestions
    
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
    tokenizer, model = load_generator()

    banned_phrases = [
        "i decided", "he decided", "she decided", "decided to"
    ]

    leakage_phrases = [
        "keep the idiom", "exactly unchanged", "paraphrase",
        "rules:", "do not explain", "return only one sentence",
        "keep the same meaning", "following sentence"
    ]

    examples = examples_map.get(normalize_idiom(idiom), [])
    candidates = [ex["en"] for ex in examples if ex.get("en")]

    if not candidates:
        return None

    random.shuffle(candidates)
    idiom_pattern = re.compile(r"\b" + re.escape(idiom) + r"\b", re.IGNORECASE)

    for base_sentence in candidates:
        try:
            prompt = f"""Paraphrase the following sentence.
Rules:
- Keep the idiom "{idiom}" exactly unchanged.
- Keep the same meaning.
- Return only one sentence.
- Do not explain anything.
Sentence:
{base_sentence}"""

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
            )
            sentence = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

            if not sentence or len(sentence.split()) < 8:
                continue
            if len(sentence.split()) < len(base_sentence.split()) * 0.6:
                continue
            if not idiom_pattern.search(sentence):
                continue
            if sentence.strip().lower() == base_sentence.strip().lower():
                continue
            if any(leak in sentence.lower() for leak in leakage_phrases):
                continue

            structure = normalize_structure(sentence)
            if structure in used_structures:
                continue
            used_structures.add(structure)

            sentence = idiom_pattern.sub("_____", sentence)
            return sentence

        except Exception:
            continue

    # ---------- FALLBACK TO DATASET ----------
    examples = examples_map.get(normalize_idiom(idiom), [])
    valid_examples = []
    for ex in examples:
        text = ex.get("en", "")
        if idiom_pattern.search(text):
            if any(bad in text.lower() for bad in banned_phrases):
                continue
            structure = normalize_structure(text)
            if structure not in used_structures:
                valid_examples.append(text)

    if valid_examples:
        sentence = random.choice(valid_examples)
        used_structures.add(normalize_structure(sentence))
        sentence = idiom_pattern.sub("_____", sentence)
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
    max_ai_attempts=5 # bef:2
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

    t0 = time.time()
    for idiom in available[:max_ai_attempts]:
        sentence = generate_ai_sentence(idiom, examples_map, used_structures)
        if not sentence:
            continue

        used_questions.add(idiom)
        wrong_pool = [i for i in idioms if i != idiom]
        wrong_options = random.sample(wrong_pool, min(3, len(wrong_pool)))
        options = wrong_options + [idiom]
        random.shuffle(options)
        print(f"[TIMING] question generated: {time.time() - t0:.2f}s")
        return {
            "question": sentence,
            "options": options,
            "answer": idiom
        }

    print(f"[TIMING] all attempts failed: {time.time() - t0:.2f}s")
    used_questions.clear()
    used_structures.clear()
    return None
    
    
