# Pipeline Diyagramı

Projenin uçtan uca veri akışı aşağıdaki Mermaid diyagramında özetlenmiştir.

## Üst düzey akış

```mermaid
flowchart LR
    subgraph Bronze["Bronze (raw)"]
        B1[CV PDF / DOCX / TXT / MD]
        B2[Job description files]
    end

    subgraph Silver["Silver (cleaned)"]
        S1[cleaned_cvs.csv]
        S2[cleaned_jobs.csv]
        SU[unified_resumes.jsonl]
    end

    subgraph Features["Feature channels"]
        F1[TF-IDF cosine]
        F2[Semantic SBERT cosine]
        F3[Skill Jaccard]
        F4[Experience match]
    end

    subgraph Scoring["Scoring & ranking"]
        SC1[Min-max normalization]
        SC2[Late fusion - weighted sum]
        SC3[Top-K ranking + explanations]
    end

    subgraph Gold["Gold (artifacts)"]
        G1[candidate_scores.csv]
        G2[candidate_scores_explained.csv]
        G3[tfidf_model.pkl]
        G4[artifacts/runs/* manifest.json]
    end

    subgraph Evaluation["Offline evaluation"]
        E1[ground_truth.csv]
        E2[Precision@K / NDCG@K / MRR / MAP]
    end

    subgraph UI["UI"]
        U1[Streamlit dashboard]
    end

    B1 -->|ingest| S1
    B2 -->|ingest| S2
    SU --> S1
    S1 --> F1
    S1 --> F2
    S1 --> F3
    S1 --> F4
    S2 --> F1
    S2 --> F2
    S2 --> F3
    S2 --> F4
    F1 --> SC1
    F2 --> SC1
    F3 --> SC1
    F4 --> SC1
    SC1 --> SC2 --> SC3
    SC3 --> G1
    SC3 --> G2
    F1 --> G3
    SC3 --> G4
    G1 --> E2
    G2 --> E2
    E1 --> E2
    G2 --> U1
```



## Skor formülü

```text
final_score = 0.35 * tfidf_score
            + 0.35 * semantic_score
            + 0.20 * skill_score
            + 0.10 * experience_score
```

Semantic kanal kapalıysa ağırlık kalan üç kanal arasında yeniden normalize edilir.

## Açıklanabilir CSV kolonları

```
job_id, cv_id, rank_for_job,
tfidf_score, semantic_score, skill_score, experience_score, final_score,
matched_skills, missing_skills, explanation
```

## Önemli komutlar


| Amaç                                         | Komut                                |
| -------------------------------------------- | ------------------------------------ |
| Bronze → Silver                              | `python main.py --ingest`            |
| Hızlı baseline (TF-IDF + skill + experience) | `python main.py --no-semantic`       |
| Tam pipeline (semantic dahil)                | `python main.py --semantic`          |
| Açık değerlendirme                           | `python main.py --evaluate`          |
| Dashboard                                    | `streamlit run app/streamlit_app.py` |


---

*Belge sürümü: 2.0 — 2026-05*