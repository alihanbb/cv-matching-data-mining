# External Dataset Import

This document describes how third-party datasets are brought into the project
as a **one-time import step** and how they align with the preferred Bronze JSONL schema.

---

## Supported Sources

| Source folder                         | `--source` alias | Notes                                         |
| ------------------------------------- | ---------------- | --------------------------------------------- |
| `NLP_NER_ON_RESUME`                   | `nlp_ner`        | Sample JSON resume text                       |
| `Entity-Recognition-In-Resumes-SpaCy` | `dataturks`      | DataTurks train/test NER annotations          |
| `vacancy-resume-matching-dataset`     | `vanetik`        | DOCX CVs + vacancy CSV, ground-truth template |
| `NER-Annotated-CVs`                   | `mehyar`         | Annotated JSON (ZIP must be extracted first)  |

Full path matching uses the `REPO_ALIASES` map in the import script.
If a folder is not found, a warning is logged and that source is skipped.

---

## Bronze Outputs

| Output file                                            | Description                                                                 |
| ------------------------------------------------------ | --------------------------------------------------------------------------- |
| `data/bronze/resumes/resumes_bronze.jsonl`             | Resume rows (`resume_id`, `raw_text`, `source`, …)                          |
| `data/bronze/jobs/jobs_bronze.jsonl`                   | Job-description rows                                                        |
| `data/bronze/annotations/ner_annotations_bronze.jsonl` | NER entity lists (profile / enrichment use)                                 |
| `*.stats.json`                                         | Row counts and `source` distribution per JSONL file                         |
| `data/evaluation/ground_truth.csv`                     | Possible template or partial ground truth (manual verification recommended) |

**Fallback:** If the JSONL files above are missing or empty, the ingest step reads raw files from:

- `data/bronze/cvs/` — PDF, DOCX, TXT, MD
- `data/bronze/job_descriptions/` — same extensions

> **Important:** External repositories are required **only during import**.
> After Bronze JSONL files are generated, the runtime pipeline reads only the
> project's standardized Bronze layer.

---

## Import Commands

Clone the four external repos into the **parent directory** of `cv-matching-data-mining`,
then run from the project root:

```bash
# Import all sources
python scripts/import_external_repos_to_bronze.py --source-root .. --all --overwrite
```

```bash
# Import a single source
python scripts/import_external_repos_to_bronze.py --source-root .. --source vanetik --overwrite
```

---

## Source Usage Strategy

| Source                   | Typical use                                                      |
| ------------------------ | ---------------------------------------------------------------- |
| **Vanetik**              | Ranking corpus + evaluation base (job–CV pairs, GT template)     |
| **DataTurks / Mehyar**   | Silver profile / NER; aligned via `ner_corpus_sources` in config |
| **NLP\_NER\_ON\_RESUME** | Schema reference; sample volume may be small                     |

`ingest.ranking_sources` in `config/config.yaml` controls which `source`-tagged rows
enter the ranking corpus (empty list = accept all sources).

---

## Limitations

- External repo layouts may change over time; field names may need updating.
- Automatic `ground_truth.csv` does not guarantee full accuracy; see `docs/GROUND_TRUTH_GUIDE.md`.
- The Mehyar dataset may require extracting a ZIP archive first.

---

## License and Ethics Notes

- Respect the license and terms of use of each third-party dataset.
- For personal data handling: `docs/KVKK_VE_GUVENLIK.md`.

---

## After Import

Standard pipeline chain after a successful import:

```bash
python main.py --ingest
python main.py --semantic --bm25
python main.py --evaluate
```

If `ground_truth.csv` is not present, evaluation is **skipped gracefully** with a log message
(`Evaluation skipped: … not found.`) and the pipeline continues without error.

---

### Optional: legacy Silver unifier (backup path)

```bash
python -m src.ingest.unify_datasets --source-root .. --output data/silver/unified_resumes.jsonl
```

**Recommended production flow:** Bronze JSONL → `python main.py --ingest` → Silver tables and profiles.
