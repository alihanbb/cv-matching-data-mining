# Veri katmanları (ideal mimari)

| Katman | Dizin | İçerik |
|--------|--------|--------|
| **Bronze** | `bronze/cvs/`, `bronze/job_descriptions/` | Ham dosyalar (PDF, DOCX, TXT, MD). İşlenmez; yalnızca ingest kaynağı. |
| **Silver** | `silver/` | Normalize tablolar: `cleaned_cvs.csv`, `cleaned_jobs.csv` (`cv_id`/`job_id`, `text`). |
| **Gold** | `gold/models/`, `gold/rankings/` | Eğitilmiş TF-IDF vektörleyici, sıralama çıktıları. |
| **Etiketler** | `evaluation/` | Offline değerlendirme: `ground_truth.csv`. |

Deney manifestleri: `artifacts/runs/<UTC>/manifest.json`.
