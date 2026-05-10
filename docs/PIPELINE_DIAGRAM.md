# Pipeline Diyagramı

Projenin uçtan uca veri akışı aşağıdaki Mermaid diyagramında özetlenmiştir.

## Üst düzey akış

```mermaid
flowchart TB
    XR["External repos<br/>(cloned sibling folders)"]

    IMP["Import to Bronze JSONL<br/><code>import_external_repos_to_bronze.py</code>"]

    BR["Bronze layer<br/>JSONL resumes, jobs, NER"]

    FB["Folder fallback<br/>bronze/cvs · job_descriptions"]

    SV["Silver cleaning & profiling<br/>CSVs · profiles · stats"]

    FE["Feature extraction<br/>TF-IDF · SBERT · BM25 · skills · experience"]

    GD["Gold ranking<br/>scores · explained · top-k"]

    EV["Evaluation<br/>optional ground_truth.csv"]

    UI["Dashboard / notebooks / Colab"]

    XR --> IMP
    IMP --> BR
    BR --> SV
    FB -.->|if JSONL missing| SV
    SV --> FE --> GD --> EV --> UI
```

## Açıklanabilir CSV (özet kolonlar)

`source`, kanal skorları, `skill_jaccard_score`, `cv_quality_score`, `must_have_coverage`, `nice_to_have_coverage`,
`final_score_v1`, `final_score_v2_bm25`, `fusion_minmax_normalized_v1`, `score_check`, `score_diff`, `score_warning`,
liste ve metinsel açıklama kolonları.

## Skor formülü (Hybrid V1, raw bileşenler)

```text
final_score_v1 ≈ 0.35 * tfidf_score + 0.35 * semantic_score + 0.20 * skill_score + 0.10 * experience_score
```

(`config/config.yaml` ile değiştirilebilir.) `ranking_score` için kanallar iş bazında min–max ile normalize edilir; ayrıca `fusion_minmax_normalized_v1` kolonu bu min–max füzyonunu raporlar.

## Hybrid V2 + BM25 (raw bileşenler)

```text
final_score_v2_bm25 =
  0.25 * tfidf + 0.25 * semantic + 0.20 * bm25 + 0.20 * skill + 0.10 * experience
```

## Önemli komutlar

| Amaç | Komut |
|------|------|
| External → Bronze JSONL | `python scripts/import_external_repos_to_bronze.py --source-root .. --all --overwrite` |
| Bronze → Silver | `python main.py --ingest` |
| Hızlı baseline (TF-IDF + structured) | `python main.py --no-semantic` |
| Tam pipeline (semantic) | `python main.py --semantic` |
| Hybrid V2 + BM25 | `python main.py --semantic --bm25` |
| Değerlendirme (GT opsiyonel) | `python main.py --evaluate` |
| Model karşılaştırma CSV | `python main.py --export-eval-csv` |
| Dashboard | `streamlit run app/streamlit_app.py` |


---

*Belge sürümü: 2.0 — 2026-05*