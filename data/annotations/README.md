# Gold annotations - 50-patient evaluation set

Single-annotator project gold used by Section 9 of `notebooks/clinical_nlp_pipeline.ipynb`. It was
built by **automatic pre-annotation** (the deterministic workflow in
`notebooks/build_gold_preannotations.ipynb`, whose raw outputs live in `generated/`) followed by
**manual row-level review**. It is an educational resource, not a certified clinical benchmark: no
independent second annotator or inter-annotator agreement was available.

Every report is identified by `encOid` / `reportOid` and a SHA-256 hash of its source text, so the
annotations can be checked against the exact input.

## Files

| File | Rows | What it annotates |
|---|---|---|
| `gold_report_manifest.csv` | 150 | The 50 evaluation patients × 3 reports, with `report_type`, `reportOid`, `text_length`, `sha256`. |
| `gold_discharge_medications_all.csv` | 378 | Discharge-therapy drug mentions: offsets, `active_ingredients`, `prescription_status`, `include_in_current_regimen`. |
| `gold_ingress_medications.csv` | 304 | Admission-therapy drug mentions: offsets, `active_ingredients`, `medication_status`. |
| `gold_anamnesis_context.csv` | 1283 | **Exhaustive** condition mentions in the 50 anamneses, with `[start, end)` offsets and three context axes (`assertion`, `temporality`, `experiencer`). |
| `gold_icd9_links.csv` | 32 | Condition → ICD-9-CM targets: `gold_status` (`code` or a NIL reason), `gold_icd9_code`, `gold_icd9_label`, `annotation_note`. |

## Annotation policy

- **Discharge / admission medications** are scored by active ingredient: discharge as a patient-level
  active-ingredient multiset (duplicates counted), admission by relaxed surface alignment of the
  normalized mention text. The pipeline outputs carry no offsets, so these are not exact-span.
- **Anamnesis conditions** use a deliberately **broad clinical-condition definition**: every disease,
  condition, symptom, abnormal-finding, infection, injury and congenital-defect mention is annotated;
  drugs, procedures, devices, normal findings and bare organisms are excluded. Spans are maximal and
  non-overlapping with zero-based `[start, end)` offsets, which lets Section 9 measure GLiNER recall
  (exact-span and relaxed IoU), not only precision.
- **Context** is annotated on three independent axes because a single label cannot represent, e.g.,
  *"Non familiarità per cardiopatie"* (negated **and** family).
- **ICD-9 targets allow `NIL`**: findings, organisms, drug names and vague fragments are marked
  not-codable instead of being forced onto an unrelated code. The `annotation_note` records the
  rationale for each decision.

## `generated/`

Raw, unreviewed outputs of the pre-annotation notebook (auto manifest, brand→ingredient lexicon,
discharge/admission auto-annotations and their review queues, a JSONL bundle, and integrity checks).
They document the deterministic bootstrap and are **not** the final gold.
