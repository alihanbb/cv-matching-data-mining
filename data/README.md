# Veri dizinleri

Bu klasör **medallion** (Bronze / Silver / Gold) düzenini izler. Büyük üretim çıktıları `.gitignore` ile repodan dışlanabilir; yapı aşağıdaki gibidir.

| Katman | Yol | İçerik |
|--------|-----|--------|
| **Bronze** | `bronze/resumes/resumes_bronze.jsonl` | Kanonik özgeçmiş satırları (`resume_id`, `raw_text`, `source`, …) |
| | `bronze/jobs/jobs_bronze.jsonl` | İlan satırları |
| | `bronze/annotations/ner_annotations_bronze.jsonl` | NER anotasyonları (profil zenginleştirme) |
| | `bronze/cvs/`, `bronze/job_descriptions/` | JSONL yoksa veya boşsa ingest fallback: PDF, DOCX, TXT, MD |
| **Silver** | `silver/cleaned_cvs.csv`, `cleaned_jobs.csv` | Sıralama matrisinde kullanılan CV / iş tabloları (temiz metin gömülü) |
| | `silver/resume_profiles.jsonl`, `job_profiles.jsonl` | Özet profiller |
| | `silver/silver_stats.json` | Satır sayıları ve üretim metrikleri |
| | `silver/unified_resumes.jsonl` | İsteğe bağlı birleşik kayıt (config’de `write_unified_resumes`) |
| **Gold** | `gold/rankings/*.csv` | Skorlar, açıklamalar, top-K |
| | `gold/models/tfidf_model.pkl` | Eğitilmiş TF-IDF |
| | `gold/evaluation/*.csv` | `--export-eval-csv` çıktıları |
| **Etiket** | `evaluation/ground_truth.csv` | Offline metrikler (zorunlu değil) |

Yerel kategori PDF arşivi (**`cv_analysis/data/data/...`**) doğrudan burada okunmaz; `scripts/import_cv_analysis_data_to_bronze.py` ile `bronze/resumes/resumes_bronze.jsonl` içine yazılır.

Teknik ayrıntı ve örnek sayılar: [docs/VERI_YAPILARI_VE_PROFILE.md](../docs/VERI_YAPILARI_VE_PROFILE.md).
