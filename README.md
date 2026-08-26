# Clinical NLP over Italian Hospital Records - a Neuro-Symbolic Pipeline

A clinical NLP pipeline over 857 pseudonymized Italian hospital encounters - each with a
discharge-therapy, an admission-therapy and an anamnesis report. Each report type gets the method its
surface form justifies: regex extraction for the templated discharge therapy, few-shot generative
extraction with a local LLM for the heterogeneous admission therapy, and zero-shot NER (GLiNER) plus
rule-based context classification (ConText) and UMLS entity linking with abstention for the free
anamnesis prose. On top of those facts sit a symbolic alert engine, an RDF knowledge graph with
SPARQL, and three local-LLM demonstrations including tool calling over the graph.

The deliverable is **`notebooks/clinical_nlp_pipeline.ipynb`** (sections 1-10, runs end-to-end).

`data/external/ingredient_crosswalk.csv` is prepared once and shipped ready to use: one row per
canonical active ingredient with its RxCUI and DDInter identifier, resolved over the **full AIFA
registry** rather than over this cohort, so the pipeline loads the identifiers instead of re-computing
the joins at run time.

## Layout

```
notebooks/
  clinical_nlp_pipeline.ipynb                   the deliverable (sections 1-10)
data/
  pazienti_con_terapia_uscita_testuale.json     dataset (857 patients x 3 reports)
  external/                                     external knowledge resources (AIFA, DDInter, MED-RT, RxNorm)
    aifa/normalized/aifa_confezioni.parquet     AIFA package registry; product -> active ingredients + ATC
    ingredient_crosswalk.csv                    one row per canonical ingredient; RxCUI and DDInter identifiers
    ddinter/normalized/safety/                  DDInter 2.0 pairwise interaction table with severity
    rx/rxclass/medrt_contraindications.csv      MED-RT ci_with relation (RxCUI + MeSH disease descriptor)
    umls_cache/                                 UMLS API results, versioned on purpose (see "UMLS access")
  annotations/                                  row-by-row gold for the 50-patient evaluation
    gold_umls_links.csv                         120-mention UMLS gold over the 30 held-out patients
outputs/                                        the GLiNER span cache, plus what a run writes
requirements.txt
.env.example                                    the one environment variable the notebook needs
```

Other files under `data/external/` are the inputs and manifests that document how the shipped
resource tables were built. The notebook itself reads only the five listed above.

### What a run writes to `outputs/`

| file | written by | holds |
|---|---|---|
| `anamnesis_entities.parquet` (+ `.meta.json`) | **Section 5** | GLiNER spans over all 857 anamnesis reports |
| `admission_mentions.parquet` (+ `.meta.json`) | **Section 4** | few-shot admission mentions with their offsets |
| `discharge_prescriptions.parquet` | **Section 3** | one row per discharge prescription |
| `conditions_linked.parquet` | **Section 5** | condition mentions with their context axes and UMLS link or abstention |
| `alerts.parquet` | **Section 6** | one row per alert, with the identifiers the rule fired on |
| `knowledge_graph.ttl` | **Section 8** | the RDF graph |

The two `.meta.json` sidecars are cache signatures, not results: each records the model and the
settings that produced the file, so changing any of them invalidates the cache instead of silently
reusing output from a different configuration. Only the GLiNER span cache ships with the repository,
because it is the one artifact whose recomputation is expensive; the rest a run rebuilds in seconds.

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
> It also matters for reproducibility: the model does the sentence segmentation GLiNER runs on, so a
> different version would re-tag all 857 reports with a different segmenter than the one behind the
> shipped span cache and the reported figures.

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

Start Ollama before the run (`ollama serve`, or the desktop app): the local model is used twice, by
the admission extraction of Section 4 and by the demonstrations of Section 9.

**How long it takes depends on the caches**, and both of them ship with the repository:

| cache | in the repository? | cost if deleted |
|---|---|---|
| `outputs/anamnesis_entities.parquet` - GLiNER spans | yes | GLiNER pass over 857 reports, ~15 min |
| `outputs/admission_mentions.parquet` - few-shot extraction | no, a run writes it | one local-LLM call per annotated patient |
| `data/external/umls_cache/` - UMLS API results | yes, 692 KB | **~1 hour** for the ~7,200 distinct condition surface forms, at the client's 4 requests/second floor |

So a fresh clone finishes in minutes. The UMLS cache holds the responses this project fetched with
its own UTS key - it is **not** the UMLS release, it carries no credential, and it is versioned on
purpose so the linking can be re-run **without a key**. A key is needed only to query surface forms
the cache has never seen. The cache is keyed by UMLS release, and searches that returned nothing are
recorded too, so an interrupted run resumes instead of starting over. Deleting a cache triggers its
recomputation, and that is the only situation in which the cold cost above applies.

**Demonstration patient.** Section 9 works on a real cohort patient, chosen deterministically. It
belongs to the patients already processed above, so the demonstrations show the path end to end and
say nothing about unseen input.

**Memory note.** GLiNER is loaded only when its span cache is missing, and the local model runs in the
Ollama process rather than in the kernel, so the notebook fits a 4 GB GPU / 16 GB RAM machine.

## Gold evaluation

Section 7 evaluates against the single-annotator project gold in `data/annotations/`, on the 30
held-out patients: **patient-level ingredient-multiset evaluation for discharge therapy, exact-span
detection for admission therapy, exact-span condition NER with and without the category, macro-F1 per
context axis** (assertion and experiencer), **UMLS entity linking** against `gold_umls_links.csv` - a
120-mention stratified sample covering all 30 held-out patients - and an alert comparison that runs
the same four rules on annotated and on extracted inputs.

The 20 development patients supply the four few-shot examples of Section 4 and nothing else. No
held-out annotation and no held-out text enters any choice the notebook makes.

> The dataset consists of pseudonymized clinical records used for an educational project only. No
> re-identification is attempted and no extensive raw text is published outside the notebook.
