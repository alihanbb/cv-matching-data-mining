# CV Matching Data Mining

Açıklanabilir, çok kanallı bir **CV–Job Description eşleştirme** sistemi.
Klasik veri madenciliği akışını (Bronze → Silver → Gold) takip eder; baseline TF-IDF
kanalını, çok dilli **SBERT** semantic kanalı, **skill** ve **experience** kanalları
ile birleştirip **late fusion** üzerinden ölçülebilir / açıklanabilir bir final skor üretir.

---

## Project goal

Bu projenin amacı, bir iş ilanı verildiğinde:

- her bir CV için **açıklanabilir bir uyum skoru** üretmek,
- adayları sıralamak,
- her skorun bileşenlerini (TF-IDF / Semantic / Skill / Experience) ve
  eşleşen / eksik becerileri göstermek,
- offline metriklerle (Precision@K, NDCG@K, MRR, MAP) kaliteyi ölçmek,
- bir **Streamlit dashboard** üzerinden insan kullanıcının sonuçları incelemesini sağlamak.

Tüm akış tekrar üretilebilir, config tabanlı ve testlerle korunur.

---

## Data mining pipeline

KDD (Knowledge Discovery in Databases) sürecine birebir oturur:

1. **Veri toplama** — Bronze: PDF / DOCX / TXT / MD CV ve iş ilanı dosyaları.
2. **Ön işleme** — Tokenize, küçük harfe çevirme, stopword temizleme, lemmatization.
3. **Bilgi keşfi** — Beceri (lexicon + alias) ve deneyim (regex) çıkarımı.
4. **Özellik çıkarımı** — TF-IDF, SBERT embedding, Skill Jaccard, Experience match.
5. **İndirgeme + Birleştirme** — Min-max normalize + ağırlıklı late fusion.
6. **Modelleme** — Top-K sıralama + açıklama üretimi.
7. **Değerlendirme** — Precision@K / NDCG@K / MRR / MAP.

Detay: [docs/RAPOR.md](docs/RAPOR.md), diyagram: [docs/PIPELINE_DIAGRAM.md](docs/PIPELINE_DIAGRAM.md).

---

## Bronze / Silver / Gold data architecture

| Katman | Dizin | İçerik |
|---|---|---|
| **Bronze** (ham) | `data/bronze/resumes/resumes_bronze.jsonl`, `data/bronze/jobs/jobs_bronze.jsonl` _(tercih edilen)_ veya klasör ingest: `data/bronze/cvs/`, `data/bronze/job_descriptions/` | Orijinal içerik; JSONL pipeline için kanonik forma dönüştürülmüş satırlar. |
| **Silver** (temiz) | `data/silver/cleaned_cvs.csv`, `data/silver/cleaned_jobs.csv`, `data/silver/unified_resumes.jsonl`, `data/silver/resume_profiles.jsonl`, `data/silver/job_profiles.jsonl`, `data/silver/silver_stats.json` | Normalize tablo ve profiller; CV’lerde `cv_id` (Bronze `resume_id`), PII maskeli metin, çıkarılmış beceriler/bölümler. |
| **Gold** (model + sonuç) | `data/gold/models/tfidf_model.pkl`, `data/gold/rankings/candidate_scores.csv`, `data/gold/rankings/candidate_scores_explained.csv`, `data/gold/rankings/top_candidates_by_job.csv`, `data/gold/evaluation/*.csv` | Eğitilmiş özellikler, sıralama ve değerlendirme çıktıları. |
| **Etiketler** | `data/evaluation/ground_truth.csv` | Dereceli relevans (0–3), offline değerlendirme. |
| **İz / Manifest** | `artifacts/runs/<UTC>/manifest.json` | Config özeti, artifact yolları, metrikler. |

## External Dataset Import

Bu proje, dış CV/NER veri kaynaklarını doğrudan proje bağımlılığı yapmaz. Aynı üst dizine klonlanan repolardaki veriler tek seferlik import scripti ile standart Bronze JSONL formatına dönüştürülür.

| Source | Purpose | Output |
|---|---|---|
| NLP_NER_ON_RESUME | Structured resume parsing reference | `resumes_bronze.jsonl` |
| Entity-Recognition-In-Resumes-SpaCy | Resume NER annotations | `resumes_bronze.jsonl`, `ner_annotations_bronze.jsonl` |
| vacancy-resume-matching-dataset | CV–job matching / evaluation | `resumes_bronze.jsonl`, `jobs_bronze.jsonl`, `ground_truth.csv` (veya template) |
| NER-Annotated-CVs | Skill/entity annotations | `resumes_bronze.jsonl`, `ner_annotations_bronze.jsonl` |

```bash
python scripts/import_external_repos_to_bronze.py --source-root .. --all --overwrite
```

Import sonrası (dış repoları silebilirsiniz):

```bash
python main.py --ingest
python main.py --semantic --bm25
python main.py --evaluate
```

Dış veri repoları yalnızca import aşamasında gereklidir. Veriler Bronze JSONL’e alındıktan sonra proje kendi standart veri katmanı üzerinden çalışır.

## Baseline model: TF-IDF + Cosine Similarity

- 1–2-gram, sublinear TF.
- CV ve iş ilanı tek vektörleyici ile fit edilir.
- Cosine similarity → `(n_cv, n_job)` matrisi.
- Skill ve experience kanalları açıklanabilirliği güçlendirir; semantic kanal kapalıdır.

```bash
python main.py --no-semantic
```

---

## Advanced hybrid model (V1): TF-IDF + SBERT + requirement coverage + experience

Kanallar:

1. **TF-IDF cosine** — lexical taban.
2. **Semantic cosine (SBERT)** — varsayılan `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (encode + L2 normalize; yüklenemezse anlaşılır uyarı).
3. **Skill score** — ilan içi *must-have / nice-to-have* bölümlerinden türetilen beceri kapsamı:
   `skill_score = 0.75 * must_have_coverage + 0.25 * nice_to_have_coverage` (nice-to-have yoksa `skill_score = must_have_coverage`).
   Yanında **skill Jaccard** kolonu tutulur (`skill_jaccard_score`).
4. **Experience** — CV ve ilan metninden İngilizce/Türkçe yıl ifadeleri + minimum gereksinim eşlemesi.

```bash
pip install -e ".[semantic]"
python main.py --semantic
```

## BM25 and Hybrid V2

- **BM25** (`rank-bm25`): her ilan için CV corpus üzerinde skor; iş bazında [0,1] min–max normalize.
- **Hybrid V2 (raw ağırlık formülü)**:

```
final_score_v2_bm25 =
  0.25 * tfidf_score +
  0.25 * semantic_score +
  0.20 * bm25_score +
  0.20 * skill_score +
  0.10 * experience_score
```

```bash
pip install -e ".[bm25]"
python main.py --semantic --bm25
```

## Weight optimization and learned fusion

```bash
python main.py --optimize-weights [--bm25]   # artifacts/best_fusion_weights.json + weight_search_results.csv
python main.py --use-best-weights --semantic [--bm25] --ingest  # sonraki koşuda ağırlıkları yükle
python main.py --train-fusion               # artifacts/learned_fusion_weights.json (PyTorch, ground truth gerekli)
```

Öğrenilen softmax-ağırlıklar mevcutsa `candidate_scores_explained.csv` içine `learned_fusion_score` yazılır.

## Cross-encoder rerank (opsiyonel)

```bash
pip install -e ".[semantic]"
python main.py --semantic --bm25
python main.py --rerank
```

Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` — her iş için ilk 20 adayı yeniden sıralar.

---

## Final score formula and auditing

**Raporlanan Hybrid V1 (`final_score_v1` / `final_score`)** — ham (normalize edilmemiş) kanal skorları üzerinden yapılandırılmış ağırlıklar:

```
final_score_v1 =
  w_tfidf * tfidf_score + w_sem * semantic_score + w_skill * skill_score + w_exp * experience_score
```

`final_score_v2_bm25`, V2 ağırlıkları ile aynı mantıkta **`bm25_score` dahil** ham toplamdır.

Varsayılan V1/V2 ağırlıkları `config/config.yaml` içindedir. **`ranking_score`**, kanalları **iş bazında min–max** normalize ederek üretilen füzyon skorudur (sıralama anahtarı).

**Min–max ile elde edilmiş ayrı gösterge** (normalize edilmiş füzyon, `final_score` kolonunun yerine geçmez):

- `fusion_minmax_normalized_v1` — V1 ağırlıkları + min–max kanal füzyonu (çift başına).

Ek denetim kolonları:

- `score_check` (V1 ağırlıklarıyla satır üzerinden yeniden hesaplanan ham toplam; `final_score_v1` ile eşleşmelidir), `score_diff`, `score_warning`

Açıklanabilir CSV öne çıkan kolonlar:

- `source`, `skill_jaccard_score`, `cv_quality_score`, `final_score_v1`, `final_score_v2_bm25`, `must_have_coverage`, `nice_to_have_coverage`, ilgili skill listeleri
- `cv_years_experience`, `job_min_years_experience`, `explanation`, `suggested_improvements`

## Requirement coverage (skill_score)

`config/skills.yaml` ile alias + kategori tabanlı skill lexicon (ör. `k8s→kubernetes`, `postgres→postgresql`, `.net→csharp`).

## PII and silver JSONL

- `privacy.anonymize: true` (varsayılan): e-posta / URL / telefon benzeri desenler skorlamadan önce maskelenir.
- `silver.write_unified_resumes: true` → `data/silver/unified_resumes.jsonl` (bölümler, skill’ler, `cv_quality_score`).

---

## Datasets used and imports

Harici paketler:

```bash
pip install -e ".[data_imports]"   # Hugging Face
pip install -e ".[kaggle_import]"  # Kaggle CLI
```

Scriptler: `scripts/import_hf_cv_matcher.py`, `scripts/import_vacancy_resume_dataset.py`, `scripts/import_kaggle_resume_dataset.py`

Ayrıntı: [docs/DATASETS.md](docs/DATASETS.md)

---

## Model comparison and evaluation export

Ground truth varken:

```bash
python main.py --export-eval-csv
```

Çıktı: `data/gold/evaluation/evaluation_results.csv`, `data/gold/evaluation/model_comparison.csv`

Karşılaştırılan modeller: TF-IDF baseline (**tfidf_score** ağırlıklı ham toplam), **Semantic Only**, Hybrid V1 (**final_score_v1** ile uyumlu ham füzyon), Hybrid V2+BM25 (**final_score_v2_bm25** ile uyumlu), Optimized Fusion (best weights dosyası varsa).

Metrikler: Precision@K, Recall@K, NDCG@K, MRR, MAP.

---

## How to run (nihai komutlar)

```bash
python main.py --ingest
python main.py
python main.py --semantic
python main.py --bm25
python main.py --evaluate
python main.py --export-eval-csv
python main.py --optimize-weights
python main.py --use-best-weights
python main.py --train-fusion
python main.py --rerank
streamlit run app/streamlit_app.py
```

> Not: `--bm25` hem anlamsal hem BM25 paketinin kurulu olmasını gerektirir.

### Setup

```bash
cd cv-matching-data-mining
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev,semantic,bm25,dashboard]"
```

### Bronze → Silver

```bash
python main.py --ingest
```

### Pipeline

```bash
python main.py --semantic        # V1 sıralama (BM25 kapalıysa)
python main.py --semantic --bm25 # V2 sıralama
python main.py --evaluate       # ground_truth ile log metrikleri
```

Çıktılar: `candidate_scores.csv`, `candidate_scores_explained.csv`, `tfidf_model.pkl`, `artifacts/runs/...`

---

## Dashboard

```bash
pip install -e ".[dashboard]"
streamlit run app/streamlit_app.py
```

Sekmeler: Candidate Ranking, CV Profile (unified JSONL), Requirement Coverage, Evaluation Metrics, Model Comparison, Score Debug.

---

## Limitations

- Skill / must-have ayrımı kural tabanlıdır; serbest metin farklı başlıklarda hata yapabilir.
- Transformer modelleri ilk indirmede ağ ve disk gerektirir; kurumsal ortamda cache önerilir.
- Kaggle / HF import scriptleri şema değişikliklerine karşı ince ayar gerektirebilir.
- Hedef NDCG@5 ikinci veri setine göre değişir; %15–25 iyileştirme iddiası deneysel olarak doğrulanmalıdır.

---

## Future work

- Ontoloji genişletmesi (ESCO vb.), aktif öğrenme ile etiket azaltma.
- Çok dilli JD başlık sınıflandırıcısı (must / nice segmentasyonu).
- Model registry ve üretim API’si + denetim günlüğü.

---

## Project layout

```
cv-matching-data-mining/
├── app/
│   └── streamlit_app.py        # Dashboard
├── config/
│   ├── config.yaml
│   └── skills.yaml             # Skill alias + kategori lexicon
├── data/
│   ├── bronze/                 # Ham dosyalar
│   ├── silver/                 # Temiz tablolar + canonical JSONL
│   ├── gold/                   # Model + sıralama çıktıları
│   └── evaluation/             # ground_truth.csv
├── docs/
│   ├── RAPOR.md
│   ├── PIPELINE_DIAGRAM.md
│   ├── KVKK_VE_GUVENLIK.md
│   └── ...
├── src/
│   ├── ingest/                 # Bronze -> Silver, unify_datasets
│   ├── preprocessing/          # TextCleaner, tokenizer
│   ├── extraction/             # Skill / experience çıkarımı
│   ├── features/               # TF-IDF, semantic encoder
│   ├── scoring/                # Fusion + explain
│   ├── models/                 # Matcher, similarity
│   ├── evaluation/             # Metrics + ranking metrics
│   └── pipeline/orchestrator.py
├── tests/                      # pytest birim testleri
├── artifacts/runs/             # Deney manifestleri (gitignore)
├── main.py
└── pyproject.toml
```

---

## Tests & CI

```bash
pytest -q
```

GitHub Actions: `.github/workflows/ci.yml` (`pytest` + `python main.py --no-semantic`).

---

## Documentation map

| Belge | Açıklama |
|--------|-----------|
| [docs/RAPOR.md](docs/RAPOR.md) | KDD süreciyle hizalı veri madenciliği raporu |
| [docs/PIPELINE_DIAGRAM.md](docs/PIPELINE_DIAGRAM.md) | Mermaid pipeline diyagramı |
| [docs/KVKK_VE_GUVENLIK.md](docs/KVKK_VE_GUVENLIK.md) | Kişisel veri ve anonimleştirme |
| [docs/GROUND_TRUTH_GUIDE.md](docs/GROUND_TRUTH_GUIDE.md) | Etiket dosyası şeması |
| [docs/MODEL_COMPARISON.md](docs/MODEL_COMPARISON.md) | Model karşılaştırma çıktıları |
| [docs/DATASETS.md](docs/DATASETS.md) | Harici veri setleri ve import |
| [docs/PROJE_KAVRAMSAL_REHBER.md](docs/PROJE_KAVRAMSAL_REHBER.md) | Senior data scientist bakışıyla genel rehber |
| [docs/CALISTIRMA_ORTAMI_VE_GELISTIRME_YONETIMI.md](docs/CALISTIRMA_ORTAMI_VE_GELISTIRME_YONETIMI.md) | Operasyon ve geliştirme yönetimi |
| [docs/MEVCUT_DURUM_VE_MIMARI.md](docs/MEVCUT_DURUM_VE_MIMARI.md) | Güncel durum ve mimari |
| [docs/YOL_HARITASI.md](docs/YOL_HARITASI.md) | Fazlı yol haritası |

---

*Versiyon: 0.3 — 2026-05*
