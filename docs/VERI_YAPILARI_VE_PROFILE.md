# Veri yapıları ve korpus profili

Bu belge, **Bronze → Silver → Gold** katmanlarında tutulan yapıların **alanları**, **kayıt mantığı**, **ölçüler** ve **tipik içerik** özeti için tek referanstır. Yerel makinedeki rakamlar bir **örnek anlık görüntüdür**; yeniden içe aktarma veya farklı `ranking_sources` seçimi ile değişir.

İlgili operasyon komutları: [`DATASETS.md`](DATASETS.md) · Mimari diyagram: [`PIPELINE_DIAGRAM.md`](PIPELINE_DIAGRAM.md) · Etiketler: [`GROUND_TRUTH_GUIDE.md`](GROUND_TRUTH_GUIDE.md)

---

## Özet: kayıt sayıları, disk ve katmanlarda ne yapıyoruz?

Bu bölüm, projede kullanılan **örnek Bronze anlık görüntüsü** (`*.stats.json` + dosya ölçüleri) için üst düzey bir özet verir.

### Toplam kayıt (Bronze düzeyi, birleşik görünüm)

| Varlık | Kayıt sayısı | Not |
|--------|---------------|-----|
| Özgeçmiş satırı (`resumes_bronze.jsonl`) | **~7750** | Örnek: harici içe aktarım + `cv_analysis_pdf_corpus` (**~2483** PDF çıkarımı) sonrası; `resumes_bronze.stats.json` |
| İş ilanı satırı (`jobs_bronze.jsonl`) | **5** | Eşleştirilecek ilan havuzu |
| NER anotasyon satırı (`ner_annotations_bronze.jsonl`) | **5202** | Varlık listeleri (profilleme / korpus için) |

Bu sayılar yeniden içe aktarma ile değişir. PDF korpus komutu ve mutlak klasör seçimi için bkz. [`DATASETS.md`](DATASETS.md).

### Veri seti boyutu (yerel kabaca disk)

Bronze’un üç ana dosyasında ölçülen düzenden:

| Dosya | Yaklaşık boyut |
|-------|-------------------|
| `resumes_bronze.jsonl` | Metin hacmine bağlı; on binlerce satır ve PDF gömülü metinle **yüzlerce MB** olabilir |
| `ner_annotations_bronze.jsonl` | ~**68 MB** |
| `jobs_bronze.jsonl` | ~**19 KB** |

Yani bu üç dosya için tipik kurulumda **yüzlerce MB** (özellikle özgeçmiş JSONL) + ilan/NER payı. Silver CSV/JSONL ve Gold sıralama CSV’leri bunlara eklenir; sıralama çalışması `n_cv × n_job` kadar çift ürettiği için Gold çıktı dosyası satır/kolon sayısı korpusa bağlı olarak büyür.

### Bronze → ne yapıyoruz?

- Dış kaynaklar (`import_external_repos_to_bronze.py`) tek tip **JSONL**’ye dönüşür.
- Üst dizindeki kategori klasörlü PDF arşivi (`scripts/import_cv_analysis_data_to_bronze.py`, varsayılan `../data/data`) aynı **`resumes_bronze.jsonl`** dosyasına **`cv_analysis_pdf_corpus`** etiketiyle eklenir.
- **`resumes/jobs`** ham metni olduğu gibi saklar (`raw_text`), kaynağı `source` ile etiketler.
- **`ner_annotations`** ayrı kümedir: metin + varlık aralıkları (beceri, isim vb.) profiler ve analiz için tutulur.
- JSONL yoksa ingest, `data/bronze/cvs/` ve `job_descriptions/` içindeki **PDF/DOCX/TXT/MD** üzerinden aynı hattı doldurur.

### Silver → ne yapıyoruz?

- **`TextCleaner`** (stopword, lemmatization, dil seçimi) ile metinleri skorlamaya uygun **`cleaned_text` / `text`** haline getiririz; istenirse **PII maskesi** (`privacy.anonymize`) uygularız.
- `config.yaml` içindeki **`ranking_sources`** dolu ise, sıralama matrisinde yalnızca bu kaynakların CV’leri kullanılır; diğer Bronze satırları profil/genişletme tarafında kalabilir.
- **`cleaned_cvs.csv` / `cleaned_jobs.csv`**: sıralama motorunun okuyacağı **CV ve iş tabloları** (matching korpus).
- **`resume_profiles.jsonl` / `job_profiles.jsonl`**: her CV/iş için lexicon tabanlı beceri özeti, bölüm uzunlukları, deneyim yılı tahmini vb. — açıklanabilirlik ve dashboard.
- İstenirse **`unified_resumes.jsonl`** (bölümle yapılmış birleşik kayıt) yazılır.
- **`silver_stats.json`** üretilen satır sayılarını özetler.

### Gold → ne yapıyoruz?

- **TF-IDF** (ve isteğe bağlı **SBERT**, **BM25**) ile kanal skorları hesaplanır; **skill** ve **experience** kanalları lexicon/regex ile beslenir.
- Kanallar **fusion** ile birleştirilir; **`candidate_scores.csv`** ve **`candidate_scores_explained.csv`** (metin açıklamaları, denetim kolonları) yazılır.
- **`top_candidates_by_job.csv`**: ilan başına en iyi adaylar.
- **`tfidf_model.pkl`**: bu Silver korpusu üzerinde eğitilmiş vektörleyici.
- **`data/gold/evaluation/`**: ground truth varsa `--export-eval-csv` ile model karşılaştırma tabloları.

---

## 1. Genel mimari ve akış

| Katman | Amaç | Proje dizinleri |
|--------|------|----------------|
| **Bronze** | Ham kaynakların kanonik JSON biçimi veya klasör fallback | `data/bronze/resumes/*.jsonl`, `jobs/*.jsonl`, `annotations/*.jsonl`, opsiyonel `cvs/`, `job_descriptions/` |
| **Silver** | Temizlenmiş sıralama tablosu, profil özeti, istatistikler | `data/silver/cleaned_*.csv`, `resume_profiles.jsonl`, `job_profiles.jsonl`, `silver_stats.json` |
| **Gold** | Modeller + sıralama / açıklama CSV’leri | `data/gold/models/`, `data/gold/rankings/`, `data/gold/evaluation/` |
| **Etiket** | Offline değerlendirme (zorunlu değil) | `data/evaluation/ground_truth.csv` |

Çalışma zamanında `main.py` **dış klon klasörlerini** ve **`cv_analysis/data/data` PDF ağacını** doğrudan okumaz; içerik import script’leriyle Bronze JSONL’ye alınır.

---

## 2. Bronze: özet şemalar

### 2.1 `resumes_bronze.jsonl`

Her satır bir JSON nesnesi.

| Alan | Tip | Açıklama |
|------|-----|----------|
| `resume_id` | string | Normalleştirilmiş CV kimliği (içeride `cv_id` ile eşlenir) |
| `source` | string | Veri kökeni (ör. `vacancy_resume_matching`, `mehyar_ner_annotated_cv`) |
| `source_file` | string | Kaynak dosya veya parça adı |
| `raw_text` | string | Ham özgeçmiş metni |
| `language` | string, isteğe bağlı | Ör. `en` |
| `labels` / `metadata` | obje, isteğe bağlı | İçe aktarılan formata göre NER / split bilgisi |

**Örnek satır sayısı dağılımı** (`data/bronze/resumes/resumes_bronze.stats.json`):

| `source` | Kayıt |
|----------|-------|
| `mehyar_ner_annotated_cv` | 4982 |
| `dataturks_resume_ner_train` | 200 |
| `dataturks_resume_ner_test` | 20 |
| `vacancy_resume_matching` | 64 |
| `nlp_ner_on_resume_json_demo` | 1 |
| `cv_analysis_pdf_corpus` | 2483 |
| **toplam (örnek)** | **7750** |

### 2.2 `jobs_bronze.jsonl`

| Alan | Açıklama |
|------|----------|
| `job_id`, `source`, `source_file`, `raw_text` | İlan kimliği ve ham metin |
| `title` | Başlık (CSV / parser’dan gelebilir) |
| `language`, `metadata` | İsteğe bağlı |

**Örnek:** bu kurulumda toplam **5** iş ilanı (`vacancy_resume_matching`), dosya yaklaşık **19 KB**.

### 2.3 `ner_annotations_bronze.jsonl`

Eğitim / servis için NER blokları tutar; ingest profil katmanına girer ama doğrudan ranking matrisinin boyutunu tek başına belirlemez.

| Alan | Açıklama |
|------|----------|
| `annotation_id` | Özgün kimlik |
| `source`, `source_file` | Köken |
| `text` | Anotasyonun dayandığı tam metin |
| `entities` | `start`, `end`, `label`, `text` listesi |
| `metadata` | `split`, biçim bilgisi vb. |

**Örnek toplam:** **5202** kayıt (DataTurks + Mehyar).

### 2.4 Dosya düzeyi boyutları (yerel örnek)

Dosya sisteminde ölçülen kabaca disk kullanımı:

| Dosya | Boyut (yaklaşık) |
|-------|------------------|
| `resumes_bronze.jsonl` | ~68 MB |
| `ner_annotations_bronze.jsonl` | ~68 MB |
| `jobs_bronze.jsonl` | ~19 KB |

---

## 3. `config.yaml` ile korpus seçimi

`ingest.ranking_sources` **dolu ise**, yalnızca listedeki `source` değerlerine sahip özgeçmişler **matching (sıralama) tablosuna** girer. Liste boşsa tüm uygun Bronze satırları kullanılabilir.

`ingest.ner_corpus_sources` ise **sıralama dışında** tutulup profil genişlemesi için kullanılan kaynakların etiketleridir (`build_processed`: bu `source` değerleri ranking listesinden düşülür).

`ingest.cv_corpus_jsonl` etkin ise `path` altındaki JSONL (`max_rows` ile sınırlı) ek CV satırı doğurabilir; yeni `cv_id`’ler ranking korpusuna eklenebilir.

Örnek (mevcut repodaki varsayılan):

- `ranking_sources`: `vacancy_resume_matching`, `sample`, `huggingface_cv_matcher`
- `cv_corpus_jsonl.max_rows`: `12000`
- `silver.write_unified_resumes`: projede sıklıkla `false`; `true` yapılırsa birleşik JSONL yazımı açılır.

---

## 4. Silver: yapılar ve içerik

### 4.1 `cleaned_cvs.csv` ve `cleaned_jobs.csv`

**Önemli:** `cleaned_cvs.csv` **tüm Bronze profillerini** değil, ingest sonrası **sıralama matrisinde kullanılacak** benzersiz `cv_id` kümesini taşır (`merged_ranking`). Metinler çok satırlı olduğundan CSV satır sayısı `wc -l` ile güvenilir ölçülmez.

Kolonlar (CV):

| Kolon | Anlamı |
|-------|--------|
| `cv_id` | Eşleştirme kimliği |
| `source` | Bronze `source` |
| `source_file` | Dosya kökeni |
| `raw_text` | Ham |
| `cleaned_text`, `text` | `TextCleaner` sonrası skorlamada kullanılan metin |

İş tablosu: `job_id`, `source`, `source_file`, `title`, `raw_text`, `cleaned_text`, `text`.

**Örnek dosya boyutları:** `cleaned_cvs.csv` ~390 KB, `cleaned_jobs.csv` ~46 KB (ham metin gömülü).

### 4.2 `resume_profiles.jsonl` ve `job_profiles.jsonl`

Satır başına özet yapı — skor kanalları ve Streamlit için hafif sinyaller.

Örnek `resume_profiles` alanları:

- `cv_id`, `source`, `source_file`
- `total_years_experience`, `cv_quality_score`, `skills_count`
- `skill_categories` (lexicon’a göre gruplanmış etiketler)
- `extracted_skill_ids_sample`
- `section_lengths` (özet, beceri, deneyim, eğitim vb. karakter uzunlukları)
- `silver_layer`: ör. `resume_profile_v1`

**Örnek `silver_stats.json` özeti:**

| Metrik | Değer |
|--------|-------|
| `resume_profiles_lines` | 5260 |
| `job_profiles_lines` | 5 |
| `unified_resumes_lines` | 5260 (üretim açıksa) |
| `profiles_dropped_short_text` | 7 |
| `privacy_anonymized` | `true` |

### 4.3 `unified_resumes.jsonl`

Bölümlere ayrılmış, beceri listeleri ve kalite skoru içeren birleşik kayıt formatı. `write_unified_resumes: true` ile ingest sırasında güncellenir; kapalıysa dosya önceki koşulardan kalabilir.

---

## 5. Gold: çıktılar ve anlamı

| Çıktı | İçerik |
|-------|--------|
| `candidate_scores.csv` | İş–CV çiftleri, kanal skorları, `ranking_score` |
| `candidate_scores_explained.csv` | Açıklama metinleri, skill kapsamları, `score_check` / `score_diff` / `score_warning`, `fusion_minmax_normalized_v1` |
| `top_candidates_by_job.csv` | İş başına top-K adaylar |
| `tfidf_model.pkl` | Eğitilmiş vektörleyici |
| `data/gold/evaluation/*.csv` | `--export-eval-csv` ile model karşılaştırma sonuçları |

**Matris boyutu:** teorik tam skor tablosu **n_cv × n_job** çiftidir; pratikte her CV her işe karşı skorlanır (bellek ve süre veri boyutuna göre büyür).

---

## 6. Ground truth ve derecelendirme

- Dosya yolu varsayılan: `data/evaluation/ground_truth.csv` (`paths.ground_truth`).
- Kolonlar ve dereceler: [`GROUND_TRUTH_GUIDE.md`](GROUND_TRUTH_GUIDE.md).
- Dosya **yoksa** pipeline düşmez; logda `Evaluation skipped: … not found.` benzeri mesaj görülür.

---

## 7. Teknik notlar ve sınırlar

- Bronze metinleri **İngilizce ağırlıklı** örnekler içerir; Türkçe ilan eklemek `./data/bronze/job_descriptions` veya uyumlu JSONL ile mümkündür.
- **`ranking_sources` daraltılmış** yapılandırmada (örnek repoda olduğu gibi) Bronze’taki çok satırlı Mehyar / DataTurks kümesi profil için tutulurken **matching listesi** daha küçük kalabilir — bu tasarım (kalite + hız) tercihidir, hata değildir.
- JSONL yazımında geçersiz UTF-16 surrogate karakterleri substitution ile güvenli hale getirilir (`external_bronze_import.write_jsonl`).
- Windows konsolunda log için Unicode oklar yerine ASCII (`->`) kullanımı önerilir (`orchestrator` / ingest CLI).

---

*Belge sürümü: 1.2 — 2026-05*
