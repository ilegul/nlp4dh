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
    umls_cache/                                 UMLS API results, versioned on purpose (see "UMLS access")
  annotations/                                  row-by-row gold for the 50-patient evaluation
    gold_umls_links.csv                         120-mention UMLS gold over the 30 held-out patients
outputs/                                        cached intermediates (GLiNER spans, parquet, the KG .ttl)
requirements.txt
.env.example                                    the one environment variable the notebook needs
```

### Which section writes which artifact

Every file in `outputs/` is prefixed with the section that writes it.

| file in `outputs/` | written by | holds |
|---|---|---|
| `sec3_discharge_drugs.parquet` | **Section 3** | one row per discharge prescription |
| `sec3_discharge_linked.parquet` | **Section 3** | one row per discharge active ingredient, with ATC / ATC-4 / RxCUI / DDInter |
| `sec4_ingress_drugs.parquet` | **Section 4** | one row per admission mention occurrence, with offsets and link provenance |
| `sec5_labelset_comparison.parquet` (+ `.meta.json`) | **Section 5A** | the GLiNER label-set ablation, on the 20 development patients |
| `sec5_anamnesis_entities.parquet` (+ `.meta.json`) | **Section 5A** | the GLiNER spans over all 857 anamnesis reports |
| `sec5_conditions_linked.parquet` | **Section 5C** | condition mentions with their context axes and UMLS link or abstention |
| `sec6_alerts.parquet` | **Section 6** | one row per alert, with rule, identifiers and provenance |
| `sec8_knowledge_graph.ttl` | **Section 8** | the RDF graph, 278,224 triples |
| `demo_synthetic_spans.json` | **Section 9** | GLiNER spans for the synthetic demo patient, kept out of every cohort figure |

The two `.meta.json` sidecars are cache signatures, not results: each records the model, the label
set, the threshold and a content hash of both the extractor source and the input texts, so changing
any of them invalidates the cache instead of silently reusing spans from an older configuration.

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

# 3) spaCy Italian model - install the PINNED version, not the current one
python -m pip install https://github.com/explosion/spacy-models/releases/download/it_core_news_lg-3.7.0/it_core_news_lg-3.7.0-py3-none-any.whl

# 4) local LLM (Section 9). Install Ollama from https://ollama.com, then:
ollama pull qwen2.5:3b-instruct
```

The two Hugging Face models (`intfloat/multilingual-e5-small`, `urchade/gliner_multi-v2.1`)
download automatically on first use.

> **Why the spaCy model is pinned.** `python -m spacy download it_core_news_lg` installs whatever is
> current - today 3.8.0, which spaCy itself warns is not compatible with the pinned `spacy==3.7.5`.
> It also matters for reproducibility: the GLiNER cache sidecars in `outputs/` record the model
> version in their signature, so an unpinned model **invalidates the cache** and re-tags all 857
> reports with a different sentence segmenter than the one behind the reported figures. Install
> 3.7.0 and the cache is reused, the run takes minutes, and the numbers reproduce.

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
excludes `.env`, `.env.*` (except `.env.example`), and any `*.RRF` or `*UMLS*.zip` release file. It
also excludes a stray `outputs/umls_cache/` or `cache/umls/`; the one cache that **is** versioned is
`data/external/umls_cache/`, deliberately, and it was checked to hold no credential.

**The full 5.4 GB UMLS release archive is not needed.** The notebook uses the REST API plus a local
cache, and reads no `MRCONSO.RRF`.

**Running without a key.** The pipeline runs without one, because `data/external/umls_cache/` ships
with the repository: a key is needed only to query a surface form the cache has never seen. Cells
that need the API load whatever the cache holds and, when a value is missing and no key is
configured, report that the step was skipped. Nothing downstream invents a concept. Only if the cache
is deleted *and* no key is configured does condition linking abstain with `api_unavailable`, and the
Section 7.5 evaluation then cannot be computed.

## Running

```bash
jupyter notebook notebooks/clinical_nlp_pipeline.ipynb   # then Run All
```

Start Ollama before the run (`ollama serve`, or the desktop app), otherwise the Section 9
demonstrations record a connection failure instead of a result.

**How long it takes depends on the caches**, and both of them ship with the repository:

| cache | in the repository? | cost if deleted |
|---|---|---|
| `outputs/` - GLiNER spans, parquet intermediates, the KG | yes | GLiNER pass ~15 min |
| `data/external/umls_cache/` - UMLS API results | yes, 692 KB | **~1 hour** for the ~7,200 distinct condition surface forms, at the client's 4 requests/second floor |

So a fresh clone finishes in minutes. The UMLS cache holds the responses this project fetched with
its own UTS key - it is **not** the UMLS release, it carries no credential, and it is versioned on
purpose so the linking can be re-run **without a key**. A key is needed only to query surface forms
the cache has never seen. The cache is keyed by UMLS release, and searches that returned nothing are
recorded too, so an interrupted run resumes instead of starting over. Deleting a cache triggers its
recomputation, and that is the only situation in which the cold cost above applies.

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
relaxed mention detection for admission therapy, condition NER split into span detection, category
on matched spans and typed NER, context evaluation on all three axes** (assertion, experiencer and
temporality - the last is reported and then gates nothing, because it scores below its own majority
baseline), **allergen extraction, and UMLS entity linking** against
`gold_umls_links.csv` - a 120-mention stratified sample covering all 30 held-out patients.

The 20 development patients are used to select the NER label set and the ConText scope, and to inspect
the numerical scale of the UMLS thresholds on their extracted mentions. No held-out annotation and no
held-out text enters any selection step.

> The dataset consists of pseudonymized clinical records used for an educational project only. No
> re-identification is attempted and no extensive raw text is published outside the notebook.
