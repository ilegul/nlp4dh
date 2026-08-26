# Clinical NLP over Italian Hospital Records

University NLP project over 857 pseudonymized Italian hospital encounters.

The pipeline applies a different method to each of the three report types:

- **Discharge therapy:** regex-based extraction from a templated report
- **Admission therapy:** few-shot medication extraction with a local LLM
- **Anamnesis:** GLiNER NER (zero-shot), ConText assertion/experiencer classification and UMLS entity linking

Condition mentions are linked to UMLS locally: a TF-IDF character n-gram index over every
Disorders alias generates candidate concepts and multilingual E5 reranks them. The extracted facts
feed four
deterministic safety rules, an RDF knowledge graph queried with SPARQL and three local-LLM
demonstrations.

## Repository

```text
notebooks/
  clinical_nlp_pipeline.ipynb
build_umls_index.py
data/
  pazienti_con_terapia_uscita_testuale.json
  annotations/
  external/
outputs/
requirements.txt
```

`data/external/` holds the AIFA, RxNorm, DDInter, MED-RT and UMLS resources the notebook reads.
`build_umls_index.py` builds the UMLS alias table and TF-IDF index from a UMLS release archive.

## Setup

Tested with Python 3.10.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux / macOS:

```bash
source .venv/bin/activate
```

Install the Python dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Install the Italian spaCy model, pinned to the version the notebook was run with:

```bash
python -m pip install https://github.com/explosion/spacy-models/releases/download/it_core_news_lg-3.7.0/it_core_news_lg-3.7.0-py3-none-any.whl
```

Install [Ollama](https://ollama.com/) and pull the local instruction model:

```bash
ollama pull qwen2.5:3b-instruct
```

GLiNER and multilingual E5 download automatically from Hugging Face on first use.

The reported run used an NVIDIA GPU with the PyTorch CUDA 12.1 build. The notebook also runs on CPU.
For the CUDA build:

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

## UMLS

Entity linking runs locally and needs no API key: candidates come from a TF-IDF index over every
UMLS Disorders alias (English and Italian), built once from the UMLS Metathesaurus full release
archive. The archive is licensed by the U.S. National Library of Medicine and is not part of this
repository: obtain a free licence, download it from the
[UTS downloads page](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html),
put it at `data/external/umls/`, then:

```bash
python build_umls_index.py
```

## Run

Start Ollama, then:

```bash
jupyter notebook notebooks/clinical_nlp_pipeline.ipynb
```

Execute the notebook from top to bottom. Generated artifacts are written to `outputs/`.


## Disclaimer

Educational project not intended for clinical use.
