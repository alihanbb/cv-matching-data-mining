# Datasets — external import and Bronze JSONL

Bu belge, üçüncü taraf veri setlerinin projeye **tek seferlik** nasıl alındığını ve tercih edilen Bronze şemasıyla nasıl hizalandığını özetler.

---

## Hedef mimari (Bronze)

| Dosya | İçerik |
|-------|--------|
| `data/bronze/resumes/resumes_bronze.jsonl` | Özgeçmiş satırları (`resume_id`, `raw_text`, `source`, …) |
| `data/bronze/jobs/jobs_bronze.jsonl` | İş ilanı satırları |
| `data/bronze/annotations/ner_annotations_bronze.jsonl` | NER varlık listeleri (profil / zenginleştirme) |

**Fallback:** Yukarıdaki JSONL dosyaları yoksa veya boşsa ingest, şu klasörlerdeki ham dosyaları okur:

- `data/bronze/cvs/` — PDF, DOCX, TXT, MD
- `data/bronze/job_descriptions/` — aynı uzantılar

Eşleştirme kodu **klonlanmış dış repo klasörlerini doğrudan kullanmaz**; dış veri `scripts/import_external_repos_to_bronze.py` ile bu Bronze yapısına taşınır.

---

## Desteklenen dış kaynaklar

| Kaynak klasörü | `--source` kısa adı | Not |
|----------------|---------------------|-----|
| `NLP_NER_ON_RESUME` | `nlp_ner` | Örnek JSON Resume metni |
| `Entity-Recognition-In-Resumes-SpaCy` | `dataturks` | DataTurks train/test NER |
| `vacancy-resume-matching-dataset` | `vanetik` | DOCX CV + vacancy CSV, GT şablonu |
| `NER-Annotated-CVs` | `mehyar` | Annotated JSON (ZIP açılmış olmalı) |

Tam yol `import` script’indeki `REPO_ALIASES` ile eşleşir; klasör yoksa uyarı verilir ve o kaynak atlanır.

---

## Bronze çıktıları

| Çıktı | Açıklama |
|-------|----------|
| `*.stats.json` | Satır sayıları / `source` dağılımı |
| `data/evaluation/ground_truth.csv` | Mümkünse şablon veya kısmi ground truth (manuel doğrulama önerilir) |

---

## Import komutları

Üst dizinde dört repo varsa, proje kökünden (`cv-matching-data-mining`):

```bash
python scripts/import_external_repos_to_bronze.py --source-root .. --all --overwrite
```

Tek kaynak:

```bash
python scripts/import_external_repos_to_bronze.py --source-root .. --source vanetik --overwrite
```

---

## Kaynak kullanım stratejisi

| Kaynak | Tipik kullanım |
|--------|----------------|
| **Vanetik** | Sıralama + değerlendirme tabanı (iş–CV eşlemesi, GT) |
| **DataTurks / Mehyar** | Silver profil / NER; `config` içindeki `ner_corpus_sources` ile hizalanır |
| **NLP_NER_ON_RESUME** | Şema referansı; örnek hacim küçük olabilir |

`ingest.ranking_sources` doluysa, yalnızca listedeki `source` etiketli satırlar ranking tablosuna girer (boş liste = tüm kaynaklar kabul).

---

## Kısıtlar

- Klon iç yapısı zamanla değişebilir; alan adları güncellenmelidir.
- Otomatik `ground_truth.csv` tam doğruluk taahhüdü değildir; bkz. `docs/GROUND_TRUTH_GUIDE.md`.
- Mehyar için ZIP’in önce açılması gerekebilir.

---

## Lisans ve etik

- Üçüncü taraf içeriklerin lisans ve kullanım koşullarına uyun.
- Kişisel veri için: `docs/KVKK_VE_GUVENLIK.md`.

---

## Import sonrası standart zincir

```bash
python main.py --ingest
python main.py --semantic --bm25
python main.py --evaluate
```

`ground_truth.csv` yoksa değerlendirme uyarı ile **atlanır**, pipeline durmaz.

---

## Opsiyonel: eski Silver birleştirici

Eski tek dosyalı birleştirme (yedek akış):

```bash
python -m src.ingest.unify_datasets --source-root .. --output data/silver/unified_resumes.jsonl
```

**Önerilen** üretim akışı: Bronze JSONL → `python main.py --ingest` → Silver tablolar ve profiller.
