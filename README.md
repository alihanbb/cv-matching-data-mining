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
| **Bronze** (ham) | `data/bronze/cvs/`, `data/bronze/job_descriptions/` | Orijinal dosyalar (PDF / DOCX / TXT / MD). Hiç değiştirilmez. |
| **Silver** (temiz) | `data/silver/cleaned_cvs.csv`, `data/silver/cleaned_jobs.csv`, `data/silver/unified_resumes.jsonl` | Normalize tablo + canonical JSONL. |
| **Gold** (model + sonuç) | `data/gold/models/tfidf_model.pkl`, `data/gold/rankings/candidate_scores.csv`, `data/gold/rankings/candidate_scores_explained.csv` | Eğitilmiş özellikler ve sıralama çıktıları. |
| **Etiketler** | `data/evaluation/ground_truth.csv` | Dereceli relevans (0–3), offline değerlendirme. |
| **İz / Manifest** | `artifacts/runs/<UTC>/manifest.json` | Config özeti, artifact yolları, metrikler. |

---

## Baseline model: TF-IDF + Cosine Similarity

- 1–2-gram, sublinear TF.
- CV ve iş ilanı tek vektörleyici ile fit edilir.
- Cosine similarity → `(n_cv, n_job)` matrisi.
- Skill ve experience kanalları açıklanabilirliği güçlendirir; semantic kanal kapalıdır.

```bash
python main.py --no-semantic
```

---

## Advanced model: TF-IDF + SBERT + Skill Score + Experience Score

Dört kanal birleştirilir:

1. **TF-IDF cosine** — lexical taban.
2. **Semantic cosine (SBERT)** — `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
3. **Skill Jaccard** — `|cv_skills ∩ job_skills| / |cv_skills ∪ job_skills|`.
4. **Experience match** — CV deneyimi vs ilan minimum gereksinimi (oranlı / belirsizde 1.0).

```bash
pip install -e ".[semantic]"
python main.py --semantic
```

---

## Final score formula

```
final_score =
  0.35 * tfidf_score +
  0.35 * semantic_score +
  0.20 * skill_score +
  0.10 * experience_score
```

Semantic kanal kapalıysa ağırlık kalan üç kanal arasında otomatik yeniden normalize edilir.

Açıklanabilir CSV şeması:

```
job_id, cv_id, rank_for_job,
tfidf_score, semantic_score, skill_score, experience_score, final_score,
matched_skills, missing_skills, explanation
```

---

## How to run

### 1. Setup

```bash
cd cv-matching-data-mining
python -m venv .venv
.venv\Scripts\activate     # Windows
# source .venv/bin/activate  # Linux / macOS
pip install -e ".[dev]"
```

Optional extras:

```bash
pip install -e ".[semantic]"    # SBERT semantic kanal
pip install -e ".[dashboard]"   # Streamlit + matplotlib
```

### 2. Bronze → Silver

```bash
python main.py --ingest
```

### 3. Pipeline

```bash
python main.py                # default (config.embeddings.enabled belirler)
python main.py --no-semantic  # hızlı baseline
python main.py --semantic     # SBERT dahil
python main.py --evaluate     # ground_truth ile metrikler
```

Çıktılar:

- `data/gold/rankings/candidate_scores.csv`
- `data/gold/rankings/candidate_scores_explained.csv`
- `data/gold/models/tfidf_model.pkl`
- `artifacts/runs/<UTC>/manifest.json`

---

## How to run dashboard

Açıklanabilir skor dosyasını görsel arayüzden incelemek için:

```bash
pip install -e ".[dashboard]"
streamlit run app/streamlit_app.py
```

Dashboard özellikleri:

- İş ilanı seçimi
- Top-N aday filtresi
- Skor bileşenleri (`tfidf_score`, `semantic_score`, `skill_score`, `experience_score`, `final_score`)
- Eşleşen / eksik beceriler
- Aday başına açıklama metni
- İlan ve CV ham metnine erişim

---

## Evaluation metrics

`data/evaluation/ground_truth.csv` formatı:

```
job_id,cv_id,relevance
job_001_python_backend,cv_001_python_backend,3
job_001_python_backend,cv_003_frontend,0
...
```

Relevance dereceleri:

- `3` — çok uygun
- `2` — uygun
- `1` — zayıf uygun
- `0` — uygun değil

Çalıştırma:

```bash
python main.py --evaluate
```

Loglanan metrikler: `topk_hit_rate_K`, `precision_at_K`, `ndcg_at_K`, `mrr`, `map`.

NDCG, dereceli relevans ile hesaplanır; binary (0/1) etiketler de geriye uyumludur.

---

## Future work

- BM25 / hybrid retrieval kanalı.
- Cross-encoder rerank (örn. `ms-marco-MiniLM`).
- Validation seti üzerinden öğrenilmiş fusion ağırlıkları (Learning-to-Rank).
- Beceri eş anlamlı / ontoloji genişletmesi (ESCO veya benzeri).
- Türkçe morfoloji desteği (lemmatizer / stemmer).
- Model registry + versiyonlu artifact’lar.
- API + auth + rate limit; KVKK uyumlu denetim günlüğü.
- Dashboard içine offline metrik karşılaştırma sekmesi.

---

## Project layout

```
cv-matching-data-mining/
├── app/
│   └── streamlit_app.py        # Dashboard
├── config/
│   └── config.yaml             # Tek doğruluk: yollar, ağırlıklar, model adı
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
| [docs/PROJE_KAVRAMSAL_REHBER.md](docs/PROJE_KAVRAMSAL_REHBER.md) | Senior data scientist bakışıyla genel rehber |
| [docs/CALISTIRMA_ORTAMI_VE_GELISTIRME_YONETIMI.md](docs/CALISTIRMA_ORTAMI_VE_GELISTIRME_YONETIMI.md) | Operasyon ve geliştirme yönetimi |
| [docs/MEVCUT_DURUM_VE_MIMARI.md](docs/MEVCUT_DURUM_VE_MIMARI.md) | Güncel durum ve mimari |
| [docs/YOL_HARITASI.md](docs/YOL_HARITASI.md) | Fazlı yol haritası |

---

*Versiyon: 0.3 — 2026-05*
