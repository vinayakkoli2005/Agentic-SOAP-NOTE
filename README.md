# Agentic SOAP Note Generation Pipeline

An end-to-end clinical NLP pipeline that converts doctor–patient audio recordings into structured **SOAP notes** (Subjective, Objective, Assessment, Plan) with medical coding (SNOMED CT, ICD-10-CM, CPT). Built as a modular agentic system — one agent per task — replicating the full pipeline from `soap-note-generation.ipynb`.

---

## Pipeline Overview

```
Audio File
    │
    ▼
┌─────────────┐
│  ASR Agent  │  Whisper-large-v3-turbo → raw transcript
└─────────────┘
    │
    ▼
┌───────────────────┐
│ Diarization Agent │  LLaMA 3.1 8B → Doctor:/Patient: labels (few-shot)
└───────────────────┘
    │
    ▼
┌───────────┐
│ NER Agent │  SPELL architecture — 6-phase extraction pipeline
└───────────┘       • LLM extraction (16-rule prompt, 3-shot)
    │               • Schema enforcement
    │               • Non-clinical filter
    │               • SPELL offset verification
    │               • Triple negation detection
    │               • VerifyNER 2nd pass
    │
    ├──────────────────────────────────┐
    ▼                                  ▼
┌───────────────────┐      ┌──────────────────────┐
│   SOAP Agent      │      │ Code Enrichment Agent │   (parallel)
└───────────────────┘      └──────────────────────┘
  • 0-shot baseline           • SNOMED CT (300+ entries)
  • K-SOAP few-shot           • ICD-10-CM (MedCodER-style
    (3-shot / 5-shot,           normalize → retrieve → rerank)
     smallest / query-aware)  • CPT codes (deterministic)
  • AI Suggestion
    │
    ▼
┌─────────────────────┐
│ Evaluation Agent    │
└─────────────────────┘
  • ROUGE-1/2/L, BLEU (1–4)
  • BERTScore (doc + sentence)
  • Keyword coverage
  • Hallucination detection (3-layer)
  • LLM-as-Judge (5-dimension rubric)
  • Entity–SOAP consistency
  • Per-section eval (S/O/A/P)
    │
    ▼
┌───────────────┐
│ Report Agent  │  → TXT + JSON + ZIP export
└───────────────┘
```

---

## Models Used

| Model | Purpose |
|-------|---------|
| `openai/whisper-large-v3-turbo` | Automatic Speech Recognition |
| `meta-llama/Meta-Llama-3.1-8B-Instruct` | Diarization, NER, SOAP generation, evaluation |
| `pritamdeka/S-PubMedBert-MS-MARCO` | Semantic embeddings (SapBERT) |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | ICD-10 / CPT reranking |

---

## Project Structure

```
Agentic-SOAP-NOTE/
├── run_pipeline.py                  # Entry point
├── pipeline/
│   ├── config.py                    # Thresholds, model IDs, feature flags
│   ├── models.py                    # Global model handles + loader
│   ├── utils.py                     # LLM wrapper, JSON parser, SPELL, negation
│   └── agents/
│       ├── asr_agent.py             # Whisper transcription
│       ├── diarization_agent.py     # Speaker diarization
│       ├── ner_agent.py             # Clinical NER (SPELL architecture)
│       ├── code_enrichment_agent.py # SNOMED CT + ICD-10-CM + CPT coding
│       ├── soap_agent.py            # SOAP / K-SOAP note generation
│       ├── evaluation_agent.py      # Automatic evaluation metrics
│       └── report_agent.py          # Report builder + export
└── soap-note-generation.ipynb       # Original notebook (source of truth)
```

---

## Setup

### 1. Install dependencies

```bash
pip install transformers sentence-transformers torch pandas \
            simple-icd-10-cm scikit-learn rouge-score
```

### 2. Set your HuggingFace token

```bash
export HF_TOKEN=your_huggingface_token_here
```

> A HuggingFace account with access to `meta-llama/Meta-Llama-3.1-8B-Instruct` is required.

---

## Usage

### Run the full pipeline on an audio file

```bash
python run_pipeline.py path/to/audio.mp3
```

### With a reference SOAP note (enables ROUGE/BLEU/BERTScore)

```bash
python run_pipeline.py path/to/audio.mp3 --reference path/to/reference.txt
```

### Specify output directory

```bash
python run_pipeline.py path/to/audio.mp3 --output_dir ./results
```

### Programmatic usage

```python
from pipeline import models
from run_pipeline import main

models.load_all_models()
paths, results = main("audio.mp3", output_dir="./results")
```

---

## Configurations

The pipeline automatically runs 4 configurations and compares them:

| Config | Description |
|--------|-------------|
| `0-shot_v1_baseline` | Pure extractive SOAP, no examples |
| `3-shot_smallest` | 3 shortest few-shot examples via KMeans |
| `3-shot_query_aware` | 3 examples most similar to the input |
| `5-shot_smallest` | 5 shortest few-shot examples via KMeans |

---

## Output

Each run produces three files in the output directory:

| File | Contents |
|------|----------|
| `Clinical_Report_<timestamp>.txt` | Human-readable report with all sections |
| `Clinical_Result_<timestamp>.json` | Structured JSON with entities, codes, metrics |
| `Clinical_Output_<timestamp>.zip` | ZIP of both files above |

The report includes: raw transcript, diarized transcript, NER entities, SOAP notes, AI suggestions, SNOMED/ICD-10/CPT codes, evaluation metrics, and a comparison table across all configs.

---

## Feature Flags (`pipeline/config.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `USE_NER_VERIFICATION` | `True` | Enable VerifyNER 2nd pass |
| `USE_CODE_VERIFICATION` | `False` | Enable per-code LLM verification |
| `USE_SELF_CONSISTENCY` | `False` | Generate N candidates, pick best |
| `SELF_CONSISTENCY_N` | `3` | Number of self-consistency candidates |

---

## Key Design Principles

- **Exact replication** — every prompt, threshold, table, and function is identical to the original notebook
- **One agent per task** — clean separation of concerns, plain Python classes with a `run()` method
- **No framework lock-in** — pure Python, no LangChain or LangGraph
- **Parallelism** — SOAP generation and code enrichment run concurrently via `ThreadPoolExecutor`
- **Extensible** — designed to be extended towards a multi-agent debate architecture (as proposed in the research paper)
