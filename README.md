---
title: English-spanish IDIOMS APP
emoji: 🚀
colorFrom: red
colorTo: red
sdk: streamlit
app_file: app.py
tags:
  - streamlit
pinned: false
short_description: Streamlit template space
---
# English–Spanish Idiom Learning App

An AI-powered web application for learning English and Spanish idioms — combining a curated idiom dataset, semantic idiom detection, literal-vs-natural translation contrast, AI-assisted adaptive quizzing, and audio pronunciation, all in one platform.

Developed as part of a Bachelor's Thesis in Telecommunications Technologies Engineering at Universidad Carlos III de Madrid (UC3M).

---

## Table of Contents

- [Motivation](#motivation)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Models and Datasets](#models-and-datasets)
- [Evaluation Results](#evaluation-results)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [License](#license)

---

## Motivation

Idioms are one of the last and hardest aspects of language fluency — even learners with strong vocabulary and grammar often default to a literal reading of idiomatic expressions, missing their intended figurative meaning. Existing language-learning platforms (Duolingo, Memrise, Busuu) treat idioms as a minor part of general vocabulary practice, with no dedicated support for differentiating idiomatic meaning **between** English and Spanish.

This project addresses that gap directly, combining NLP techniques (machine translation, semantic similarity, controlled text generation) with an interactive learning interface.

## Features

| Mode | Description |
|---|---|
| **Explore** | Browse all 119 idioms by topic or search; view natural meaning, English meaning, literal translation, and audio pronunciation for each |
| **Idioms in Sentences (Detection)** | Paste free-form text and detect idioms within it — works across verb tenses and across English/Spanish input |
| **Quiz** | AI-assisted, adaptive fill-in-the-blank questions, prioritizing idioms the user has struggled with |
| **Favorites** | Bookmark idioms for quick review, persisted across sessions |
| **Learning Analytics** | Per-idiom accuracy tracking — mastered / practiced / weak breakdown, driving the adaptive quiz engine |

## Architecture

```
User (Web browser)
        │
        ▼
Streamlit UI (app.py)
  Explore · Detect · Quiz · Favorites · Analytics
        │
        ▼
Application Logic (utils.py)
  Orchestrates AI models and data
        │
   ┌────┴────┐
   ▼         ▼
AI Models   Data Layer
```

**AI Models:** MarianMT (translation) · FLAN-T5-base (quiz generation) · Multilingual MiniLM (semantic idiom detection)

**Data Layer:** SQLite (favorites, learning analytics) · `idiom2.json` (curated dataset) · Hugging Face Datasets (example sentences)

### Idiom detection pipeline (two-stage)

1. **Stage 1 — `detect_idioms`:** literal and lemma-aware matching (e.g., "broke the ice" → "break the ice"). Runs first; fast and highly precise.
2. **Stage 2 — `detect_idioms_ai`:** semantic fallback using MiniLM sentence embeddings, only consulted when Stage 1 finds no match (e.g., "good luck in your exam" → *break a leg*).

## Tech Stack

- **Frontend/App framework:** [Streamlit](https://streamlit.io/)
- **NLP/ML:** [Transformers](https://github.com/huggingface/transformers), [Sentence-Transformers](https://www.sbert.net/), [Hugging Face Datasets](https://github.com/huggingface/datasets)
- **Text-to-speech:** [gTTS](https://gtts.readthedocs.io/)
- **Lemmatization:** [lemminflect](https://github.com/bjascob/LemmInflect)
- **Storage:** SQLite
- **Deployment:** Hugging Face Spaces

## Getting Started

### Prerequisites

- Python 3.10+ (developed and tested on Python 3.13)
- Internet access on first run (see [note below](#note-on-first-run-downloads))

### Installation

```bash
git clone https://github.com/kawaa099-dot/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt --break-system-packages
```

### Running locally

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

## Project Structure

```
.
├── app.py                  # Streamlit entry point — UI and mode switching
├── utils.py                 # Application logic: model loading, dataset handling,
│                             #   detection, translation, quiz generation
├── idiom2.json               # Hand-curated dataset: 119 idioms with Spanish/English
│                             #   meanings, topic, and (partial) gloss fields
├── idioms.db                 # SQLite database (created automatically on first run)
├── audio_cache/               # Cached gTTS audio files (created automatically)
├── requirements.txt
└── README.md
```

## Models and Datasets

| Component | Source | Role |
|---|---|---|
| Translation | [`Helsinki-NLP/opus-mt-en-es`](https://huggingface.co/Helsinki-NLP/opus-mt-en-es) (MarianMT) | Literal English→Spanish translation |
| Quiz generation | [`google/flan-t5-base`](https://huggingface.co/google/flan-t5-base) | Paraphrases dataset example sentences into quiz questions |
| Idiom detection (semantic fallback) | [`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) | Embedding-based similarity matching |
| Example sentences | [`fdelucaf/IdioTS`](https://huggingface.co/datasets/fdelucaf/IdioTS), [`UCSC-Admire/idiom-SFT-dataset-561-2024-12-06_00-40-30`](https://huggingface.co/datasets/UCSC-Admire/idiom-SFT-dataset-561-2024-12-06_00-40-30) | Real-world example sentences for idioms with available coverage |
| Audio pronunciation | `gTTS` | On-demand English/Spanish audio |

All models and datasets are pulled automatically on first run — no manual download step required.

### Why three separate models?

Each task has a different shape: translation is a well-defined encoder-decoder task MarianMT specializes in; quiz generation needs controlled paraphrasing, which FLAN-T5's instruction-tuning supports; detection needs semantic similarity scoring, which a sentence-embedding model is architecturally suited for. FLAN-T5-base was chosen over GPT-4 (cost, external dependency) and Mistral-7B (too computationally demanding for CPU-only deployment) as the best balance of generation quality and deployability.

## Evaluation Results

Full methodology and discussion are in the accompanying thesis (Chapter 5). Summary:

**MarianMT literal translation** (50-idiom sample): 90% literal meaning preserved, 86% grammatically correct.

**FLAN-T5 quiz generation** (100 generated sentences): 93% valid, 95% grammatically correct, 88% usable overall; covered 22/27 idioms with available examples; 71.3% of distractors matched the correct answer's topic (vs. 0% expected by random chance).

**Idiom detection** (100-sentence labeled test set, threshold = 0.3):

| Category | n | Precision | Recall |
|---|---|---|---|
| Exact (literal match) | 17 | 1.00 | 1.00 |
| Tense (inflected) | 17 | 1.00 | 0.88 |
| Semantic (AI fallback) | 28 | 0.71 | 0.43 |
| **Overall** | | **0.733** | **0.710** (F1 = 0.721) |

Negative-sentence false positive rate: 8.70%. Hard-negative false positive rate: 60.00% — a structural limitation of the literal-matching stage, unaffected by similarity threshold tuning (see thesis Table 5.10).

**User evaluation** (6 respondents, native Spanish speakers): 66.7% found the interface clear and organized, 83.3% agreed AI-generated quizzes reinforce learning, mean motivation to continue learning: 3.67/5. Idiom detection was the lowest-rated feature (3.50/5), consistent with its lower semantic recall.

## Limitations

- Only 27 of 119 idioms have example sentences available, limiting AI-assisted quiz generation coverage to ~22.7% of the dataset.
- English–Spanish only; no support for other language pairs.
- Requires internet access on first run (or after a Hugging Face Spaces free-tier instance restarts) to download models and datasets.
- CPU-only deployment constrains model size and generation speed.
- The semantic detection fallback has a high false-positive rate (60%) on hard-negative sentences (idiom-related vocabulary used non-idiomatically) — an architectural limitation of running literal matching before semantic matching, not fixable via threshold tuning.
- User evaluation sample size (n=6) is small and drawn from the advisor's academic colleagues rather than the app's target learner demographic.

## Future Work

- Expand the dataset's example-sentence coverage beyond 27 idioms.
- User accounts to persist progress across devices.
- Automatic speech recognition for speaking-based practice.
- Personalized recommendations / streak mechanisms for spaced repetition.
- Support for additional language pairs.
- Offline mode for limited-connectivity use.

## Reproducibility

### Note on first-run downloads

Model and dataset downloads are cached **per running instance**, not per user or device. On Hugging Face Spaces' free tier, a fresh download is triggered whenever the container restarts after going idle (free tier does not persist storage between sleep cycles); all requests to an already-running instance reuse the cached copies regardless of user or device.

### Reproducing the reported evaluation results

**a) MarianMT translation quality:** 50 idioms sampled from `idiom2.json`; each `translate_literal()` output manually scored on literal-meaning preservation and grammatical correctness, verified by a native Spanish speaker.
*(Add the path to your scored sample here, e.g. `eval/marianmt_eval.csv`.)*

**b) FLAN-T5 quiz generation quality:** 100 sentences generated via `generate_ai_sentence()` through the deployed app; scored for successful generation, validity, grammatical correctness, final usability, idiom coverage, and distractor topic-matching.
*(Add the path to your generated/scored sample here, e.g. `eval/quiz_generation_eval.csv`.)*

**c) Idiom detection precision/recall/F1:** 100-sentence labeled test set across 5 categories (exact, tense-inflected, semantic, negative, hard-negative). Run `detect_idioms()` / `detect_idioms_ai()` against the labeled set and compare to ground truth to reproduce Table 5.10 (threshold = 0.3 used in reported results).
*(Add the path to your labeled test set and scoring script here, e.g. `eval/detection_test_set.csv`.)*

> **To finish this section:** add the actual file paths/scripts used to generate each evaluation sample, so results can be re-run exactly rather than only read as methodology.

### Known reproducibility caveats

- FLAN-T5 quiz generation uses sampling (`do_sample=True`), so exact generated sentences vary between runs even with identical inputs — the evaluation methodology is designed to be reproducible in aggregate (validity/grammar/usability rates), not sentence-for-sentence.
- Results were obtained on CPU-only Hugging Face Spaces infrastructure; generation latency will differ on other hardware.

## Citation

If you use this dataset or code, please cite:

```
Mohd Sukki, K. J. (2026). Design and Implementation of an AI-Powered
English-Spanish Idiom Learning System. Bachelor's Thesis, Universidad
Carlos III de Madrid.
```

## Author

**Kawtharul Jannah Mohd Sukki**
Telecommunications Technologies Engineering, Universidad Carlos III de Madrid
[GitHub](https://github.com/kawaa099-dot) · [Hugging Face](https://huggingface.co/kawaa099)
# Welcome to Streamlit!

Edit `/src/app.py` to customize this app to your heart's desire. :heart:

If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).
