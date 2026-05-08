# Veri setleri ve içe aktarma

## Proje içi CV korpusu (Silver ingest)

`config/config.yaml` → `ingest.cv_corpus_jsonl`:

- `enabled: true` ise ingest sırasında **ilave** CV satırları belirtilen JSONL’den okunur.
- **İş ilanları** yalnızca `data/bronze/job_descriptions/` altındaki dosyalardan üretilmeye devam eder.
- Varsayılan yol: `data/silver/unified_resumes.jsonl` (`record_id` + `text`; `max_rows` ile üst sınır).

```bash
python main.py --ingest
```

JSONL yoksa uyarı loglanır; yalnızca bronze CV dosyaları kullanılır.

## Hugging Face

| Dataset | Script |
|---------|--------|
| JeremiahOnu/cv-matcher-data | `python scripts/import_hf_cv_matcher.py` |
| NataliaVanetik/vacancy-resume-matching-dataset | `python scripts/import_vacancy_resume_dataset.py` |

Gerekli: `pip install -e ".[data_imports]"`

Dataset şemaları zamanla değişebilir; script içindeki alan adları (`resume_text`, `job_description`, vb.) gerekirse güncellenmelidir.

## Kaggle

`python scripts/import_kaggle_resume_dataset.py --dataset <owner/name>`

Gerekli: `pip install -e ".[kaggle_import]"` ve `~/.kaggle/kaggle.json`.

## Ground truth

İçe aktarma scriptleri mümkün olduğunda `data/evaluation/ground_truth.csv` örneği oluşturur; üretim kalitesi için manuel doğrulama önerilir.
