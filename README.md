# CV Matching Data Mining Project

## Amaç

CV ile iş ilanları arasında **çok kanallı skorlama** (lexical TF-IDF, isteğe bağlı çok dilli gömüler, beceri Jaccard’ı, deneyim uyumu) ve **açıklanabilir** sıralama üretmek.

## Yöntemler

- Metin ön işleme (stopword, lemmatization)
- TF-IDF + kosinüs
- Opsiyonel: `sentence-transformers` ile yoğun gömü (ör. çok dilli MiniLM)
- Yapılandırılmış: lexicon tabanlı beceri çıkarımı + ilan için tahmini yıl gereksinimi
- **Late fusion** (sütun bazlı min–max, ağırlıklı birleşim)
- Offline metrikler: Top-K isabet, precision@K, **NDCG@K**, **MRR**, **MAP**

## Kurulum

```bash
cd cv-matching-data-mining
pip install -e ".[dev]"
```

Yoğun gömü kanalı (ilk indirme büyük olabilir):

```bash
pip install -e ".[semantic]"
```

## Çalıştırma

```bash
# Tam pipeline (gömü yüklüyse çok dilli kanal açık)
python main.py

# Sadece TF-IDF + yapılandırılmış kanallar (CI / hızlı test)
python main.py --no-semantic

# Bronze → Silver (PDF/DOCX/TXT/MD)
python main.py --ingest
python -m src.ingest
```

Çıktılar (Gold + deney izi):

- `data/gold/rankings/candidate_scores.csv` — skor bileşenleri ve sıra
- `data/gold/rankings/candidate_scores_explained.csv` — eşleşen/eksik beceriler, deneyim notu
- `data/gold/models/tfidf_model.pkl` — eğitilmiş TF-IDF
- `artifacts/runs/<UTC>/manifest.json` — config özeti, girdi hash’leri, metrikler

## Veri katmanları

Özet tablo: `data/README.md`.

| Katman | Konum |
|--------|--------|
| Bronze (ham) | `data/bronze/cvs/`, `data/bronze/job_descriptions/` |
| Silver (işlenmiş tablolar) | `data/silver/cleaned_cvs.csv`, `cleaned_jobs.csv` |
| Gold (model + sıralama) | `data/gold/models/`, `data/gold/rankings/` |
| Etiketler (offline eval) | `data/evaluation/ground_truth.csv` |

## Yapılandırma

`config/config.yaml`: ingest yolları, TF-IDF, **fusion ağırlıkları**, embedding model adı, `top_k`, deney manifesti, log seviyesi.

## Güvenlik

Özet: `docs/KVKK_VE_GUVENLIK.md`.

## Dokümantasyon (uzun vadeli plan ve mimari)

| Belge | Açıklama |
|--------|-----------|
| [docs/MEVCUT_DURUM_VE_MIMARI.md](docs/MEVCUT_DURUM_VE_MIMARI.md) | Güncel yapı, yığın, veri akışı, bilinen sınırlar |
| [docs/YOL_HARITASI.md](docs/YOL_HARITASI.md) | Fazlı geliştirme planı ve öncelikler |
| [docs/GELISTIRME_VE_SURDURULEBILIRLIK.md](docs/GELISTIRME_VE_SURDURULEBILIRLIK.md) | Süreç, test, sürümleme, PR kontrol listesi |
| [docs/PROJE_DEGERLENDIRME_VE_IDEAL_MIMARI.md](docs/PROJE_DEGERLENDIRME_VE_IDEAL_MIMARI.md) | Tasarım gerekçesi ve ideal mimari karşılaştırması |
| [docs/KVKK_VE_GUVENLIK.md](docs/KVKK_VE_GUVENLIK.md) | Kişisel veri ve güvenlik notları |
| [data/README.md](data/README.md) | Bronze / Silver / Gold dizin sözleşmesi |
| [docs/reports/](docs/reports/) | Teslim raporları (pipeline şeması, geri besleme analizi) |

## Test ve CI

```bash
pytest -q
```

GitHub Actions: `.github/workflows/ci.yml` (pytest + `python main.py --no-semantic`).
