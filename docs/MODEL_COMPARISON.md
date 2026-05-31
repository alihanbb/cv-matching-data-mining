# Model karşılaştırması

Veri tabloları ve kolon bağlamı: [VERI_YAPILARI_VE_PROFILE.md](VERI_YAPILARI_VE_PROFILE.md).

`python main.py --export-eval-csv` komutu `data/gold/evaluation/evaluation_results.csv` ve `model_comparison.csv` üretir.

## Karşılaştırılan varyantlar

1. **TF-IDF Baseline** — yalnızca lexical kanal (ham ağırlıklı toplam).
2. **Semantic Only** — yalnızca dense/semantic kanalı (SBERT kapalıysa skorlar sıfıra yakın olur).
3. **Hybrid V1** — `config.yaml` içindeki `fusion.weights` ile **ham** kanal skorlarının ağırlıklı toplamı (`final_score_v1` ile aynı yapı).
4. **Hybrid V2 + BM25** — `fusion_v2.weights` ve BM25 (`rank-bm25`) ile **ham** toplam (`final_score_v2_bm25` ile aynı yapı); BM25 yoksa V1’e düşer.
5. **Optimized Fusion** — `artifacts/best_fusion_weights.json` mevcutsa grid search sonucu ağırlıklar.

## Opsiyonel sonrası

- `python main.py --rerank` — cross-encoder ile top-20 yeniden sıralama (dosya düzeyinde skor kolonları güncellenir; export script şu an ana fusion varyantlarını raporlar).

## Okuma listesi

- NDCG@K — dereceli uygunluk için birincil gösterge.
- Precision@K / Recall@K — kesik gözlem; iş başına etiket sayısına duyarlıdır.
