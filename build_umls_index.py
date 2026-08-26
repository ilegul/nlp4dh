"""Build the local UMLS lexical index used for entity-linking candidate generation.

Run once. The licensed release archive is read in place; the
standard build writes the files the notebook loads directly:

    data/external/umls_index/umls_concepts.parquet          one row per indexed alias
    data/external/umls_index/umls_tfidf_vectorizer.joblib   the fitted character n-gram vectorizer
    data/external/umls_index/umls_tfidf_matrix.npz          one sparse row per alias, same order

The optional --embeddings flag additionally encodes every alias with E5 into umls_embeddings.npy
(1.4 GB), used only when producing the dense-retrieval baseline cache; the operational linker
never reads it.

Aliases carry a tier, kept as metadata for analysis:

    tier 0   the preferred label of the concept, per language (MRCONSO ISPREF, MRRANK)
    tier 1   up to two synonyms per concept from the sources listed in SYNONYM_SOURCES
    tier 2   every other atom

Neither the archive nor the index may be committed: both are licensed UMLS content.

    python build_umls_index.py
"""
import argparse
import io
import zipfile
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).parent
ARCHIVE = PROJECT_ROOT / "data" / "external" / "umls" / "umls-2026AA-metathesaurus-full.zip"
INDEX_DIR = PROJECT_ROOT / "data" / "external" / "umls_index"
UMLS_VERSION = "2026AA"

EMBEDDER_MODEL = "intfloat/multilingual-e5-small"
LANGUAGES = ("ENG", "ITA")

# The semantic types of the Disorders group. Restricting to them reproduces the
# semanticGroups=Disorders filter the UTS search applied server-side, so the two 
# retrieval routes are compared over the same concepts.
DISORDERS_TUIS = {"T019", "T020", "T033", "T037", "T046", "T047", "T048",
                  "T049", "T050", "T184", "T190", "T191"}

# Synonyms are taken from these sources only. They are the vocabularies with 
# curated clinical synonymy.
SYNONYM_SOURCES = {"SNOMEDCT_US", "MSH", "MDR", "NCI", "MSHITA", "MDRITA", "ICPCITA"}
MAX_SYNONYMS_PER_CONCEPT = 2

BATCH_SIZE = 1024


def read_rrf(archive, member):
    """Yield the pipe-separated fields of each line, streamed straight from the archive."""
    with archive.open(f"{UMLS_VERSION}/META/{member}") as handle:
        for line in io.TextIOWrapper(handle, encoding="utf-8"):
            yield line.rstrip("\n").split("|")


def read_semantic_types(archive):
    """(CUIs in the Disorders group, their semantic type names) from MRSTY."""
    types = defaultdict(list)

    for fields in read_rrf(archive, "MRSTY.RRF"):
        cui, tui, name = fields[0], fields[1], fields[3]

        if tui in DISORDERS_TUIS:
            types[cui].append(name)

    return types


def read_source_ranks(archive):
    """MRRANK as a (source, term type) -> precedence lookup."""
    return {(fields[1], fields[2]): int(fields[0]) for fields in read_rrf(archive, "MRRANK.RRF")}


def read_labels(archive, semantic_types, source_ranks):
    """Every non-suppressed English or Italian atom of a Disorders concept, ranked within its CUI."""
    rows = []

    for fields in read_rrf(archive, "MRCONSO.RRF"):
        cui, language, source, term_type = fields[0], fields[1], fields[11], fields[12]
        label, suppress = fields[14], fields[16]
        is_preferred = fields[6] == "Y"

        if language not in LANGUAGES or suppress != "N" or cui not in semantic_types:
            continue

        rows.append((cui, label, language, is_preferred, source_ranks.get((source, term_type), 0),
                     source in SYNONYM_SOURCES))

    labels = pd.DataFrame(rows, columns=["cui", "label", "language", "is_preferred", "rank",
                                         "curated_source"])
    labels["semantic_types"] = labels.cui.map(lambda c: "; ".join(semantic_types[c]))
    return labels


def assign_tiers(labels):
    """Tier each label, so the notebook can narrow the index by filtering rather than rebuilding."""
    # One label per concept per language survives as tier 0: an atom MRCONSO marks as preferred,
    # with MRRANK's source precedence breaking the tie between several preferred atoms.
    labels = labels.sort_values(["cui", "language", "is_preferred", "rank"],
                                ascending=[True, True, False, False])
    labels = labels.drop_duplicates(["cui", "label"])
    labels["tier"] = 2

    preferred = ~labels.duplicated(["cui", "language"])
    labels.loc[preferred, "tier"] = 0

    synonyms = labels[~preferred & labels.curated_source]
    keep = synonyms.groupby("cui").head(MAX_SYNONYMS_PER_CONCEPT).index
    labels.loc[keep, "tier"] = 1

    return (labels.drop(columns=["is_preferred", "rank", "curated_source"])
            .sort_values(["tier", "cui"])
            .reset_index(drop=True))


def embed_labels(labels, destination):
    """Encode every label once into a matrix on disk, in the row order of `labels`.

    Vectors are L2-normalised at encoding time, so the notebook's similarity search is a dot
    product and never has to normalise anything again.
    """
    embedder = SentenceTransformer(EMBEDDER_MODEL)
    width = embedder.get_embedding_dimension()
    matrix = np.lib.format.open_memmap(destination, mode="w+", dtype=np.float16,
                                       shape=(len(labels), width))

    for start in range(0, len(labels), BATCH_SIZE):
        batch = labels.label.iloc[start:start + BATCH_SIZE]
        matrix[start:start + len(batch)] = embedder.encode(
            [f"passage: {text}" for text in batch],
            normalize_embeddings=True, show_progress_bar=False)

        if start % (BATCH_SIZE * 200) == 0:
            print(f"   {start:,} / {len(labels):,} labels encoded", flush=True)

    matrix.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    parser.add_argument("--embeddings", action="store_true",
                        help="also build the embedding matrix used when producing the "
                             "dense-retrieval baseline cache; not needed by the operational linker")
    arguments = parser.parse_args()

    arguments.index_dir.mkdir(parents=True, exist_ok=True)
    archive = zipfile.ZipFile(arguments.archive)

    print("reading MRSTY: concepts of the Disorders group", flush=True)
    semantic_types = read_semantic_types(archive)
    print(f"   {len(semantic_types):,} concepts", flush=True)

    print("reading MRRANK: source precedence", flush=True)
    source_ranks = read_source_ranks(archive)

    print("reading MRCONSO: English and Italian labels", flush=True)
    labels = assign_tiers(read_labels(archive, semantic_types, source_ranks))
    print(f"   {len(labels):,} labels over {labels.cui.nunique():,} concepts", flush=True)
    print(labels.groupby(["tier", "language"]).size().rename("labels").to_frame(), flush=True)

    labels.to_parquet(arguments.index_dir / "umls_concepts.parquet", index=False)

    print("fitting the TF-IDF index", flush=True)
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)
    matrix = vectorizer.fit_transform(labels.label.str.lower())
    joblib.dump(vectorizer, arguments.index_dir / "umls_tfidf_vectorizer.joblib")
    sparse.save_npz(arguments.index_dir / "umls_tfidf_matrix.npz", matrix)
    print(f"   vocabulary {len(vectorizer.vocabulary_):,} | {matrix.nnz:,} nonzeros", flush=True)

    if arguments.embeddings:
        print(f"encoding with {EMBEDDER_MODEL} (dense baseline only)", flush=True)
        embed_labels(labels, arguments.index_dir / "umls_embeddings.npy")

    print(f"\nwritten to {arguments.index_dir}")


if __name__ == "__main__":
    main()
