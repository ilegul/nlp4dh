# Clinical NLP over Italian Hospital Records - a Neuro-Symbolic Pipeline

An end-to-end clinical NLP pipeline over 857 pseudonymized Italian hospital encounters - each with a
discharge-therapy, an admission-therapy and an anamnesis report - that
demonstrates a broad range of NLP techniques - regex extraction, a symbolic AIFA gazetteer, dense retrieval for entity linking, zero-shot
NER (GLiNER), dependency parsing, symbolic vs. neural context classification, entity linking, a
symbolic alert engine, neuro-symbolic tool-calling (Ollama), and an RDF knowledge graph with SPARQL.

The main deliverable is **`notebooks/clinical_nlp_pipeline.ipynb`**. Two supporting notebooks prepare
inputs the pipeline then just consumes: **`notebooks/build_gold_preannotations.ipynb`** (the deterministic
pre-annotation workflow that bootstraps the evaluation gold) and **`notebooks/build_drug_resources.ipynb`**
(builds two drug resources: `data/external/aifa_products.csv`, one row per AIFA product, and
`data/external/ingredient_crosswalk.csv`, one row per canonical active ingredient carrying ATC / RxCUI /
DDInter id, resolved over the full AIFA registry rather than this cohort - so the main pipeline loads the
codes instead of re-computing the joins at run time, and a drug absent from these patients is still covered).

## Layout

```
notebooks/
  clinical_nlp_pipeline.ipynb          the deliverable (11 sections, 0-10, runs end-to-end)
  build_gold_preannotations.ipynb      deterministic pre-annotation workflow (support)
  build_drug_resources.ipynb           builds the two drug resources (support)
data/
  pazienti_con_terapia_uscita_testuale.json   dataset (857 patients × 3 reports)
  external/                            external knowledge resources (AIFA, DDInter, MeSH, MED-RT, RxNorm, ICD-9-CM)
    aifa_products.csv                  one row per AIFA product; product-name catalog and product-label ATC retrieval index
    ingredient_crosswalk.csv           one row per canonical AIFA active ingredient; ATC, RxCUI and DDInter mappings with provenance
    aifa/normalized/medications.csv    raw AIFA anagrafica farmaci (builder input)
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

# 2) Python packages (PyTorch pulls the CUDA 12.1 build - see the note in requirements.txt)
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

**Support notebooks / build order.** The repository ships the prepared artifacts (`data/external/aifa_products.csv`,
`data/external/ingredient_crosswalk.csv` and the `outputs/` caches), so `clinical_nlp_pipeline.ipynb` runs
top-to-bottom out of the box. To rebuild the drug resources from scratch: run `build_drug_resources.ipynb`
(it reads the raw AIFA `aifa/normalized/medications.csv` + `aifa_confezioni.parquet`, the pre-crawled RxNav
table and the DDInter drug list, and writes `data/external/aifa_products.csv` and
`data/external/ingredient_crosswalk.csv`), then Run-All `clinical_nlp_pipeline.ipynb`. The builder no
longer depends on the main pipeline. `build_gold_preannotations.ipynb` is independent and only documents how the gold in
`data/annotations/generated/` was bootstrapped.

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
