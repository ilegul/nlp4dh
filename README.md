# Clinical NLP over Italian Hospital Records — a Neuro-Symbolic Pipeline

NLP4DH exam project. An end-to-end pipeline over 857 pseudonymized Italian discharge records that
demonstrates a broad range of course techniques — regex extraction, dense retrieval / RAG, zero-shot
NER (GLiNER), dependency parsing, symbolic vs. neural context classification, entity linking, a
symbolic alert engine, neuro-symbolic tool-calling (Ollama), and an RDF knowledge graph with SPARQL.

The main deliverable is **`notebooks/clinical_nlp_pipeline.ipynb`**. A supporting notebook,
**`notebooks/build_gold_preannotations.ipynb`**, documents the deterministic pre-annotation workflow
used to bootstrap the evaluation gold.

## Layout

```
notebooks/
  clinical_nlp_pipeline.ipynb          the deliverable (10 sections, runs end-to-end)
  build_gold_preannotations.ipynb      deterministic pre-annotation workflow (support)
data/
  pazienti_con_terapia_uscita_testuale.json   dataset (857 patients × 3 reports)
  external/                            external knowledge resources (AIFA, DDInter, MED-RT, RxNorm, ICD-9-CM)
  annotations/                         row-by-row gold for the 50-patient evaluation (see annotations/README.md)
    generated/                         auto pre-annotations produced by the support notebook
outputs/                               cached intermediates (embeddings, parquet, the KG .ttl)
requirements.txt
```

## Setup

Tested with **Python 3.10** on Windows (Intel CPU + GTX 1650, 4 GB VRAM).

```bash
# 1) virtual environment
python -m venv .venv
# Windows:            .venv\Scripts\activate
# Linux / macOS:      source .venv/bin/activate

# 2) Python packages (PyTorch pulls the CUDA 12.1 build — see the note in requirements.txt)
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3) spaCy Italian model (not a pip package)
python -m spacy download it_core_news_lg

# 4) local LLM (Step 7). Install Ollama from https://ollama.com, then:
ollama pull qwen2.5:3b-instruct
```

The three Hugging Face models (`intfloat/multilingual-e5-small`, `urchade/gliner_multi-v2.1`,
`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`) download automatically on first use.

## Running

```bash
jupyter notebook notebooks/clinical_nlp_pipeline.ipynb   # then Run All
```

The heavy stages (dictionary/ICD-9 embeddings, GLiNER over 857 anamneses, condition linking) are
**cached to `outputs/`**, so a run that finds the caches finishes in a few minutes. Deleting a cache
triggers its recomputation (the GLiNER pass is ~15 min the first time).

**Memory note.** The notebook is tuned for a 4 GB GPU / 16 GB RAM machine: GLiNER is skipped when its
cache exists, models are freed and reloaded between stages, and the Ollama model is unloaded before the
evaluation. If you have more RAM/VRAM these frees are simply harmless.

## Gold evaluation

Section 9 evaluates against the single-annotator project gold in `data/annotations/`:
**patient-level ingredient-multiset evaluation for discharge therapy, relaxed mention detection for
admission therapy, exact/relaxed span evaluation for GLiNER, multi-axis context evaluation
(assertion / temporality / experiencer), and NIL-aware ICD-9 linking**. See
`data/annotations/README.md` for the file inventory and the annotation policy.

> The dataset consists of pseudonymized clinical records used for an educational project only. No
> re-identification is attempted and no extensive raw text is published outside the notebook.
