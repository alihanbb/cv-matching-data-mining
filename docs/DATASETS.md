# Veri setleri ve içe aktarma

## Dış kaynakların Bronz’a tek seferlik aktarımı (önerilen)

Dış veri kaynakları proje içine doğrudan dahil edilmemiştir. Veriler standart Bronze JSONL şemasına dönüştürülerek pipeline’ın geri kalanının tek tip veri formatı üzerinden çalışması sağlanmıştır.

Aynı üst dizinde dört kaynak klasörünü (`NLP_NER_ON_RESUME`, `Entity-Recognition-In-Resumes-SpaCy`, `vacancy-resume-matching-dataset`, `NER-Annotated-CVs`) bir kez klonlayıp sonra **yalnızca import sırasında** şu komutu çalıştırın:

```bash
python scripts/import_external_repos_to_bronze.py --source-root .. --all --overwrite
```

| Kaynak klasörü | Kısa kod | Pipeline’ın okuduğu çıktılar |
|---|---|---|
| NLP_NER_ON_RESUME | `--source nlp_ner` | `data/bronze/resumes/` — tek JSON Resume demo |
| Entity-Recognition-In-Resumes-SpaCy | `--source dataturks` | `resumes_bronze.jsonl` + `ner_annotations_bronze.jsonl` |
| vacancy-resume-matching-dataset | `--source vanetik` | `resumes`, `jobs`, mümkünse `ground_truth.csv` |
| NER-Annotated-CVs | `--source mehyar` | `resumes_bronze` + `ner_annotations_bronze` |

Özet olarak her repo **hangi amaçla** kullanılmıştır: Vanetik = ilan–CV eşlemesi ve değerlendirme; DataTurks + Mehyar = NER / beceri çıkarımı için anotasyon örnekleri; NLP_NER = yapılandırılmış özgeçmiş metin düzleştirme referansı. **Dosyalar** import scriptinin içinde listelenmiştir. **Bronze çıktıları** şu dosyalardır: `resumes_bronze.jsonl`, `jobs_bronze.jsonl`, `ner_annotations_bronze.jsonl`, istatistik `*.stats.json`.

**Kayıt sayıları**: `*.stats.json` dosyalarına bakın — sabit sayı yazılmaz.

**Sınırlılıklar**: Klon içeriği zamanla değişebilir; otomatik ground truth çıkarımı bazı düzen dosyaları için düşecektir; `docs/GROUND_TRUTH_GUIDE.md` ve `ground_truth_template.csv` ile tamamlama gerekir.

**Lisans / etik not**: Üçüncü taraf verilerin lisansına uyun; bu depo üçüncü taraf hammaddesini sürekli taşımak yerine kullanıcı import eder.

Import sonrası **ingest** standart Bronze JSONL’yi okur (`config.yaml` içindeki `bronze_resumes_jsonl` / `bronze_jobs_jsonl`); dış klasörleri silebilirsiniz:

```bash
python main.py --ingest
```

Opsiyonel: `ingest.ranking_sources` ile yalnızca belirli `source` etiketli CV satırlarının sıralamaya dahil edilmesi (örn. Vanetik) daraltılabilir; boş liste tüm Bronze kaynaklarını dahil eder.

## Proje içi CV korpusu (Silver ingest ek satırlar)

`config/config.yaml` → `ingest.cv_corpus_jsonl`:

- `enabled: true` ise ingest sırasında **ilave** CV satırları belirtilen JSONL’den okunur.
- **İş ilanları** artık öncelikle `data/bronze/jobs/jobs_bronze.jsonl` varken ondan, yoksa `data/bronze/job_descriptions/` altından üretilir.
- Varsayılan yol: `data/silver/unified_resumes.jsonl` (`record_id` + `text`; `max_rows` ile üst sınır). `ranking_sources` dolu ise bu JSONL’deki satırlar da aynı etiket filtresine tabidir.

```bash
python main.py --ingest
```

Bronze `resumes_bronze.jsonl` / `jobs_bronze.jsonl` yoksa ingest, `data/bronze/cvs` ve `data/bronze/job_descriptions` altındaki ham dosyalara düşer. `cv_corpus_jsonl` kapalı veya dosyası eksik ise yalnızca bu ana kaynaklar kullanılır.

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

## Aynı üst dizinde klonlanan repolar (`cv_analysis`) — eski akış

`python -m src.ingest.unify_datasets` hâlâ Silver `unified_resumes.jsonl` üretmek için kullanılabilir; **önerilen** veri yolu artık `scripts/import_external_repos_to_bronze.py` ile Bronze JSONL ve ardından `python main.py --ingest` akışıdır.

| Repo | Not |
|------|-----|
| [vacancy-resume-matching-dataset](https://github.com/NataliaVanetik/vacancy-resume-matching-dataset) | Import scripti DOCX + `5_vacancies.csv` okur; `stage_cloned_repos_bronze.py` yalnızca ham kopya için kalabilir. |
| [Entity-Recognition-In-Resumes-SpaCy](https://github.com/DataTurks-Engg/Entity-Recognition-In-Resumes-SpaCy) | `traindata.json` / `testdata.json` |
| [NER-Annotated-CVs](https://github.com/Mehyarmlaweh/NER-Annotated-CVs) | `extracted/ResumesJsonAnnotated/*.json` |
| [NLP_NER_ON_RESUME](https://github.com/minhquan23102000/NLP_NER_ON_RESUME) | `resume.json` |

**NER-Annotated-CVs** için ZIP’i bir kez açın (örnek):

`Expand-Archive .../NER-Annotated-CVs/ResumesJsonAnnotated.zip -DestinationPath .../NER-Annotated-CVs/extracted`

Ardından (opsiyonel) eski birleştirici:

```bash
python -m src.ingest.unify_datasets --source-root .. --output data/silver/unified_resumes.jsonl
```

Klon klasörlerini dahil etmek istemezseniz: `--no-workspace-clones`.

Büyük NER korpusu eklendikten sonra `config/config.yaml` içinde `ingest.cv_corpus_jsonl.max_rows` değerini artırmanız gerekebilir.

## Ground truth

İçe aktarma scriptleri mümkün olduğunda `data/evaluation/ground_truth.csv` örneği oluşturur; üretim kalitesi için manuel doğrulama önerilir.
