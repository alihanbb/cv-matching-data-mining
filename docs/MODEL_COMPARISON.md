# Model karşılaştırması

`python main.py --export-eval-csv` komutu `data/gold/evaluation/evaluation_results.csv` ve `model_comparison.csv` üretir.

## Karşılaştırılan varyantlar

1. **TF-IDF Baseline** — yalnızca lexical kanal (semantic kapalı matris yapısı).
2. **TF-IDF + SBERT** — tfidf ve semantic eşit ağırlıklı basit birleşim.
3. **Hybrid V1** — `config.yaml` içindeki `fusion.weights` (skill coverage + experience dahil).
4. **Hybrid V2 + BM25** — `fusion_v2.weights` ve BM25 (`rank-bm25`); BM25 yoksa V1’e düşer.
5. **Optimized Fusion** — `artifacts/best_fusion_weights.json` mevcutsa grid search sonucu ağırlıklar.

## Opsiyonel sonrası

- `python main.py --rerank` — cross-encoder ile top-20 yeniden sıralama (dosya düzeyinde skor kolonları güncellenir; export script şu an ana fusion varyantlarını raporlar).

## Okuma listesi

- NDCG@K — dereceli uygunluk için birincil gösterge.
- Precision@K / Recall@K — kesik gözlem; iş başına etiket sayısına duyarlıdır.
