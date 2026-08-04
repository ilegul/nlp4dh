# Clinical NLP over Italian Hospital Records - a Neuro-Symbolic Pipeline

A clinical NLP pipeline over 857 pseudonymized Italian hospital encounters - each with a
discharge-therapy, an admission-therapy and an anamnesis report - that
demonstrates a broad range of NLP techniques - regex extraction, a symbolic AIFA gazetteer, zero-shot
NER (GLiNER), rule-based context classification (ConText), UMLS entity linking with abstention, a
symbolic alert engine, neuro-symbolic tool-calling (Ollama), and an RDF knowledge graph with SPARQL.

The deliverable is **`notebooks/clinical_nlp_pipeline.ipynb`** (sections 0-10, runs end-to-end).

Two drug resources are prepared once and shipped ready to use: `data/external/aifa_products.csv`, one
row per AIFA product, and `data/external/ingredient_crosswalk.csv`, one row per canonical active
ingredient carrying its ATC / RxCUI / DDInter identifiers. They are resolved over the **full AIFA
registry** rather than over this cohort, so the pipeline loads the codes instead of re-computing the
joins at run time, and a drug absent from these patients is still covered.

## Layout

```
notebooks/
  clinical_nlp_pipeline.ipynb                   the deliverable (sections 0-10)
data/
  pazienti_con_terapia_uscita_testuale.json     dataset (857 patients x 3 reports)
  external/                                     external knowledge resources (AIFA, DDInter, MED-RT, RxNorm)
    aifa_products.csv                           one row per AIFA product; product-name catalog with product-level ATC
    ingredient_crosswalk.csv                    one row per canonical AIFA ingredient; ATC, RxCUI, DDInter mappings
    aifa/normalized/medications.csv             raw AIFA anagrafica farmaci (builder input)
    umls_cache/                                 UMLS API results, NOT in git (see "UMLS access")
  annotations/                                  row-by-row gold for the 50-patient evaluation
    gold_umls_links.csv                         120-mention UMLS gold over the 30 held-out patients
outputs/                                        cached intermediates (GLiNER spans, parquet, the KG .ttl)
requirements.txt
.env.example                                    the one environment variable the notebook needs
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

# 4) local LLM (Section 9). Install Ollama from https://ollama.com, then:
ollama pull qwen2.5:3b-instruct
```

The two Hugging Face models (`intfloat/multilingual-e5-small`, `urchade/gliner_multi-v2.1`)
download automatically on first use.

## UMLS access

Section 5 links condition mentions to the UMLS Metathesaurus through the UTS REST API, which needs a
**free UMLS licence and a personal API key**: request one at <https://uts.nlm.nih.gov/uts/signup-login>,
then copy it from your UTS profile page.

The notebook reads the key **only** from the environment variable `UMLS_API_KEY`. It is never
hard-coded, printed, written to a cache, or included in an error message.

```powershell
# Windows - persistent
setx UMLS_API_KEY "your-key-here"
```

```bash
# Linux / macOS - add to ~/.bashrc or ~/.zshrc to persist
export UMLS_API_KEY="your-key-here"
```

> **After `setx` you must restart Jupyter (or the IDE that launches it).** A running process keeps the
> environment it was started with, so a kernel opened before you set the variable will not see it. To
> check from inside the notebook: the setup cell prints `API key configured: True`.

`.env.example` records the variable name and holds no value. Never commit a real key: `.gitignore`
excludes `.env`, `.env.*` (except `.env.example`), any `umls_cache/` directory wherever it sits, and
any `*.RRF` or `*UMLS*.zip` release file.

**The full 5.4 GB UMLS release archive is not needed.** The notebook uses the REST API plus a local
cache, and reads no `MRCONSO.RRF`.

**Running without a key.** The pipeline still runs: cells that need the API load whatever the local
cache holds and, when a value is missing and no key is configured, report that the step was skipped.
Nothing downstream invents a concept. Without either a key or a cache, condition linking abstains with
`api_unavailable` and the Section 7.5 evaluation cannot be computed.

## Running

```bash
jupyter notebook notebooks/clinical_nlp_pipeline.ipynb   # then Run All
```

Start Ollama before the run (`ollama serve`, or the desktop app), otherwise the Section 9
demonstrations record a connection failure instead of a result.

**How long it takes depends on the caches**, which are not all in git:

| cache | in the repository? | cost when cold |
|---|---|---|
| `outputs/` - GLiNER spans, parquet intermediates, the KG | yes | GLiNER pass ~15 min |
| `data/external/umls_cache/` - UMLS API results | **no** (licensed content) | **~1 hour** for the ~7,200 distinct condition surface forms, at the client's 4 requests/second floor |

So a run on a machine that already has both caches finishes in minutes; a **fresh clone with a valid
key pays the cold UMLS pass once**, after which the cache makes every later run fast. The cache is
keyed by UMLS release, and searches that returned nothing are recorded too, so an interrupted run
resumes instead of starting over. Deleting a cache triggers its recomputation.

**Demo patient.** The Section 9 end-to-end demonstration uses one synthetic patient, defined and tagged
inside the notebook: its anamnesis is a literal in Section 9 and its GLiNER spans come from the cohort's
own `extract_entities`, cached to `outputs/demo_synthetic_spans.json` under a signature (label set,
threshold, hash of the text) exactly like the cohort spans. Its spans live in `demo_anam_spans` and are
never merged into `anamnesis_entities`, so no corpus-level figure can see them.

**Memory note.** The notebook is tuned for a 4 GB GPU / 16 GB RAM machine: GLiNER is skipped when its
cache exists, models are freed and reloaded between stages, and the sentence embedder and spaCy are
freed before the local LLM section loads. With more RAM/VRAM these frees are simply harmless.

## Gold evaluation

Section 7 evaluates against the single-annotator project gold in `data/annotations/`, on the 30
held-out patients: **patient-level ingredient-multiset evaluation for discharge therapy, exact and
relaxed mention detection for admission therapy, exact/relaxed span evaluation for GLiNER, context
evaluation on two axes** (assertion and experiencer; temporality is extracted and used by the
contraindication rule, but not scored), **allergen extraction, and UMLS entity linking** against
`gold_umls_links.csv` - a 120-mention stratified sample covering all 30 held-out patients.

The 20 development patients are used to select the NER label set and the ConText scope, and to inspect
the numerical scale of the UMLS thresholds on their extracted mentions. No held-out annotation and no
held-out text enters any selection step.

> The dataset consists of pseudonymized clinical records used for an educational project only. No
> re-identification is attempted and no extensive raw text is published outside the notebook.
