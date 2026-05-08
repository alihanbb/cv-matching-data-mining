# Ground truth rehberi

## Dosya

`data/evaluation/ground_truth.csv`

## Şema

```text
job_id,cv_id,relevance
job_001,cv_001,3
job_001,cv_002,2
```

Kolon adı `relevance` veya `relevant` olabilir (ikincisi iç uyumluluk için `relevant` olarak normalize edilir).

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
