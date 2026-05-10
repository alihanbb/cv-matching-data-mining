# Datasets — external import & Bronze JSONL

Bu belge, projeye **dış veri setlerinin tek seferlik** nasıl dahil edildiğini ve Bronze şemasıyla nasıl hizalandığını özetler.

---

## External Dataset Import

- Dış CV/NER/job eşleştirme repoları **proje köküne gömülmez**; aynı üst dizinde veya `--source-root` altında klonlanır.
- `scripts/import_external_repos_to_bronze.py` yalnızca bu aşamada dış klasör formatlarını bilir; **`main.py` ve eşleştirme kodu yalnızca Bronze JSONL** okur.

---

## Supported Sources

| Kaynak klasörü | Kısa seçenek (`--source`) | Amaç |
|----------------|---------------------------|------|
| `NLP_NER_ON_RESUME` | `nlp_ner` | Yapılandırılmış özgeçmiş JSON referansı |
| `Entity-Recognition-In-Resumes-SpaCy` | `dataturks` | DataTurks export (train/test NER) |
| `vacancy-resume-matching-dataset` | `vanetik` | CV–job DOCX + vacancy CSV, eşleştirme GT |
| `NER-Annotated-CVs` | `mehyar` | Annotated JSON korpusu |

Tam yol `import` script’i içindeki `REPO_ALIASES` ile eşleştirilir; klasör bulunamazsa **warning** yazılır ve o kaynak atlanır.

---

## Bronze Outputs

| Çıktı dosyası | Açıklama |
|---------------|-----------|
| `data/bronze/resumes/resumes_bronze.jsonl` | Normalize CV satırları (`resume_id`, `raw_text`, `source`, …) |
| `data/bronze/jobs/jobs_bronze.jsonl` | Normalize iş ilanı satırları |
| `data/bronze/annotations/ner_annotations_bronze.jsonl` | NER varlık dizileri |
| `*.stats.json` | Satır sayıları / `source` dağılımı |
| `data/evaluation/ground_truth.csv` | Mümkün olduğunda şablon veya kısmi GT (manuel doğrulama önerilir) |

---

## Import Commands

Üst dizinde dört repo klonlu iken, proje kökünden (`cv-matching-data-mining`):

```bash
python scripts/import_external_repos_to_bronze.py --source-root .. --all --overwrite
```

Tek kaynak:

```bash
python scripts/import_external_repos_to_bronze.py --source-root .. --source vanetik --overwrite
```

---

## Source Usage Strategy

| Kaynak | Tipik kullanım |
|--------|----------------|
| **Vanetik** | Ranking + değerlendirme (iş–CV eşleşmesi, GT üretimi için taban) |
| **DataTurks / Mehyar** | Silver profil / NER zenginleştirme; config’teki `ner_corpus_sources` ile hizalanır |
| **NLP_NER_ON_RESUME** | Metin çıkarım şeması referansı; örnek hacmi küçük olabilir |

`config/config.yaml` içinde **`ingest.ranking_sources`** dolu ise, yalnızca listedeki `source` etiketli satırlar ranking tablosuna girer (boş liste = tüm kaynaklar ranking için uygun varsayılır).

---

## Limitations

- Klonların iç yapısı zamanla değişebilir; script’ler alan adı değişimlerinde güncellenmelidir.
- Otomatik `ground_truth.csv` tam ve hatasız olmayabilir; `docs/GROUND_TRUTH_GUIDE.md` ile doğrulama önerilir.
- DOCX/ZIP çıkarımı kullanıcı ortamına bağlıdır (ör. Mehyar ZIP’inin önce açılması gerekebilir).

---

## License and Ethics Notes

- Üçüncü taraf içeriklerin **lisans ve kullanım koşullarına uyun**.
- Kişisel veri içeren korpuslar için KVKK / GDPR uyumlu süreç: `docs/KVKK_VE_GUVENLIK.md`.

---

## After Import

Bronze oluşturulduktan sonra standart zincir:

```bash
python main.py --ingest
python main.py --semantic --bm25
python main.py --evaluate
```

Opsiyonel: eski tek dosyalı Silver birleştirici (yedek akış):

```bash
python -m src.ingest.unify_datasets --source-root .. --output data/silver/unified_resumes.jsonl
```

**Önerilen** akış: Bronze JSONL → `python main.py --ingest` → Silver tablolar / profiller.
