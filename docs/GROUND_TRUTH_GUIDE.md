# Ground truth rehberi

## Dosya

`data/evaluation/ground_truth.csv`

Kolonlar: `job_id`, `resume_id` (alternatif olarak `cv_id`), `relevance` (veya iç kullanımda `relevant`) ve isteğe bağlı `source`.

`vacancy-resume-matching-dataset` anotasyon dosyası otomatik parse edilemezse `data/evaluation/ground_truth_template.csv` oluşturulur; içeriği kopyalayıp `relevance` sütununu manuel tamamlayın ve `ground_truth.csv` olarak kaydedin.

Örnek (Vanetik):

```text
job_id,resume_id,relevance,source
vanetik_vacancy_001,vanetik_cv_001,3,vacancy_resume_matching
```

## Şema

```text
job_id,cv_id,relevance
job_001,cv_001,3
job_001,cv_002,2
```

Kolon adı `relevance` veya `relevant` olabilir (`relevant`’e normalize edilir). Satırlarda `resume_id` varsa dahili olarak `cv_id` olarak okunur.

## Dereceler

| Değer | Anlam |
|-------|--------|
| 3 | Çok uygun |
| 2 | Uygun |
| 1 | Zayıf uygun |
| 0 | Uygun değil |

## Kullanım

- `python main.py --evaluate` — pipeline sonunda Precision@K, Recall@K, NDCG@K, MRR, MAP loglanır.
- `python main.py --export-eval-csv` — `data/gold/evaluation/` altına model karşılaştırma CSV’leri.
- `python main.py --optimize-weights` / `--train-fusion` — etiketli çiftler gerektirir; ID’ler `cleaned_cvs.csv` / `cleaned_jobs.csv` ile birebir eşleşmelidir.

## İpuçları

- Aynı iş için birden fazla aday satırı verin; NDCG için derece dağılımı faydalıdır.
- Bronze dosya adlarından türetilen `cv_id` / `job_id` ile tutarlılığa dikkat edin (`--ingest` çıktısını kontrol edin).
