# Pipeline Diyagramı

Projenin uçtan uca veri akışı aşağıdaki Mermaid diyagramında özetlenmiştir.

## Üst düzey akış

```mermaid
flowchart LR
    subgraph External["External repositories (import only)"]
        XR[NLP_NER_ON_RESUME / vacancy-matching / NER CVs ...]
    end

    subgraph Bronze["Bronze (canonical JSONL or folder ingest)"]
        BX[resumes_bronze.jsonl · jobs_bronze.jsonl · ner_annotations_bronze.jsonl]
        B1[CV PDF / DOCX / TXT / MD fallback]
        B2[Job description files fallback]
    end

    subgraph Silver["Silver (cleaned + profiles)"]
        S1[cleaned_cvs.csv]
        S2[cleaned_jobs.csv]
        SU[unified_resumes.jsonl · resume_profiles.jsonl · job_profiles.jsonl · silver_stats.json]
    end

    subgraph Features["Feature extraction"]
        F1[TF-IDF cosine]
        F2[Semantic SBERT cosine]
        F2b[BM25 optional]
        F3[Requirement coverage + skill Jaccard]
        F4[Experience match · cv_quality from Silver]
    end

    subgraph Gold["Gold (ranking + evaluation)"]
        G1[candidate_scores.csv · candidate_scores_explained.csv · top_candidates_by_job.csv]
        GE[evaluation_results.csv · model_comparison.csv · score_audit_report.csv]
        G3[tfidf_model.pkl]
        G4[artifacts/runs manifest]
    end

    subgraph Evaluation["Offline evaluation"]
        E1[ground_truth.csv optional]
        E2[Precision@K / Recall@K / NDCG@K / MRR / MAP]
    end

    subgraph UI["Consumption"]
        U1[Dashboard / notebooks / Colab]
    end

    XR -->|scripts/import_external_repos_to_bronze.py| BX
    BX -->|main.py --ingest priority| S1
    BX --> S2
    B1 -->|folder ingest if no JSONL| S1
    B2 --> S2
    SU --> S1
    S1 --> F1
    S1 --> F2
    S1 --> F2b
    S1 --> F3
    S1 --> F4
    S2 --> F1
    S2 --> F2
    S2 --> F2b
    S2 --> F3
    S2 --> F4
    F1 --> G1
    F2 --> G1
    F2b --> G1
    F3 --> G1
    F4 --> G1
    F1 --> G3
    G1 --> GE
    E1 --> E2
    GE --> E2
    G1 --> U1
```



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

`source`, `tfidf_score`, `semantic_score`, `bm25_score`, `skill_jaccard_score`, `skill_score`, `experience_score`, `cv_quality_score`, `must_have_coverage`, `final_score_v1`, `final_score_v2_bm25`, `fusion_minmax_normalized_v1`, `explanation`, `suggested_improvements`, ...

## Önemli komutlar


| Amaç                                         | Komut                                |
| -------------------------------------------- | ------------------------------------ |
| Bronze → Silver                              | `python main.py --ingest`            |
| Hızlı baseline (TF-IDF + skill + experience) | `python main.py --no-semantic`       |
| Tam pipeline (semantic dahil)                | `python main.py --semantic`          |
| Hybrid V2                                    | `python main.py --semantic --bm25`   |
| Açık değerlendirme                           | `python main.py --evaluate`          |
| Model karşılaştırma CSV                      | `python main.py --export-eval-csv`   |
| Dashboard                                    | `streamlit run app/streamlit_app.py` |


---

*Belge sürümü: 2.0 — 2026-05*