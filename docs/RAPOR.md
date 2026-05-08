# Veri Madenciliği Süreci — Proje Raporu

Bu rapor, **cv-matching-data-mining** projesindeki uçtan uca veri madenciliği sürecini açıklar. Yapı, klasik **KDD (Knowledge Discovery in Databases)** akışına oturtulmuş ve projedeki kod modülleriyle eşlenmiştir.

---

## 1. Veri Toplama (Data Collection)

Proje birden fazla kaynaktan veri kabul eder:


| Kaynak                  | Klasör                              | İçerik                                                |
| ----------------------- | ----------------------------------- | ----------------------------------------------------- |
| Bronze CV'ler           | `data/bronze/cvs/`                  | PDF, DOCX, TXT, MD formatında ham CV dosyaları        |
| Bronze iş ilanları      | `data/bronze/job_descriptions/`     | PDF/DOCX/TXT/MD formatında ilan dosyaları             |
| Birleşik canonical veri | `data/silver/unified_resumes.jsonl` | Çoklu açık veri setlerini tek formatta toplayan JSONL |
| Etiketli değerlendirme  | `data/evaluation/ground_truth.csv`  | Aday–ilan dereceli relevans (0–3)                     |


Sistem, ham veriyi değiştirmez; bronze katmanı kaynak bütünlüğünü korumaktan sorumludur.

İlgili modül: `src/ingest/build_processed.py`, `src/ingest/text_extract.py`, `src/ingest/unify_datasets.py`.

---

## 2. Ön İşleme (Preprocessing)

CV ve iş ilanı metinleri analiz edilebilir hale getirilir.

İşlem sırası:

1. Metin çıkarımı (PDF/DOCX/TXT/MD).
2. Küçük harfe çevirme.
3. Noktalama temizleme (ancak `+` ve `#` gibi teknolojik işaretler korunur).
4. Çoklu boşlukları normalize etme.
5. Tokenizasyon.
6. İsteğe bağlı stopword temizliği (NLTK).
7. İsteğe bağlı lemmatization (WordNet).

İlgili modül: `src/preprocessing/cleaner.py`, `src/preprocessing/tokenizer.py`.

Kalite kapısı: `src/schemas/documents.py` — Pydantic ile alan ve tip doğrulaması.

---

## 3. Bilgi Keşfi (Knowledge Discovery)

Veri içinden işe yarar sinyaller çıkarılır.


| Sinyal        | Yöntem                                                | Çıktı                                 |
| ------------- | ----------------------------------------------------- | ------------------------------------- |
| Beceri kümesi | `config/skills.yaml` lexicon + alias | CV ve ilan için kanonik `skill_id` kümesi |
| İlan gereksinimleri | Başlık anahtar kelimeleri (must / nice, TR/EN) | `must_have` / `nice_to_have` kümeleri |
| Kapsam skoru | İstenen becerilere göre \|eşleşme\| / \|gerekli\| | `skill_score` kanalı |
| Deneyim yılı  | Regex tabanlı yıl ifadeleri (EN/TR)            | CV maks. yıl, ilan minimum gereksinim     |
| Rol ipuçları  | Regex (junior, senior, lead, ...)                     | İsteğe bağlı bağlam sinyali           |


İlgili modüller: `src/extraction/skill_extractor.py`, `src/extraction/skills_lexicon.py`, `src/extraction/requirements_extractor.py`, `src/extraction/experience_extractor.py`.

Bu adım, "ham metin → yapılandırılmış sinyal" geçişini sağlar ve sonraki skor kanalları için temel oluşturur.

---

## 4. Özellik Çıkarımı (Feature Extraction)

Hesaplanan kanallar:

1. **Lexical (TF-IDF)** — `src/features/tfidf_vectorizer.py`
2. **Semantic (SBERT)** — `src/features/semantic_encoder.py`
3. **BM25 (opsiyonel)** — `src/features/bm25_scorer.py` (`rank-bm25`)
4. **Skill** — gereksinim kapsamı + raporlama için Jaccard (`src/scoring/fusion.py`)
5. **Experience** — `src/scoring/fusion.py::experience_match_matrix`

---

## 5. İndirgeme ve Birleştirme (Reduction & Late Fusion)

Her kanal `(n_cv, n_job)` boyutlu bir matris üretir. Skorlar farklı ölçeklerde olduğundan **iş bazlı min–max normalizasyonu** uygulanır, sonra ağırlıklı toplama yapılır.

Final skor formülü:

```
final_score =
  0.35 * tfidf_score +
  0.35 * semantic_score +
  0.20 * skill_score +
  0.10 * experience_score
```

Semantic kanal kapalıysa ağırlık otomatik olarak yeniden normalize edilir (kalan kanallar üstüne dağıtılır).

İlgili modüller: `src/scoring/fusion.py`, `src/scoring/explain.py`, `src/models/matcher.py`.

Açıklanabilirlik:

- Eşleşen beceriler (`matched_skills`)
- Eksik beceriler (`missing_skills`)
- Deneyim notu (`explanation`)

---

## 6. Modelleme (Modeling)

Bu projedeki "model", parametre öğrenen tek bir sınıflandırıcı değildir; **hibrit ve açıklanabilir bir skor birleşim modelidir**.

- TF-IDF + cosine: lexical taban.
- Sentence-Transformers: semantik genelleme.
- Skill Jaccard: yapılandırılmış kontrol.
- Experience match: hard/soft constraint.
- Late fusion: ağırlıklı birleşim.

İleri aşama (yol haritası):

- BM25 / hybrid retrieval.
- Cross-encoder rerank.
- Learning-to-Rank (validation seti üzerinden ağırlık öğrenme).
- Beceri eş anlamlı sözlüğü ve ontoloji.

---

## 7. Değerlendirme (Evaluation)

Çıktılar gold katmanına yazılır:


| Dosya                                               | İçerik                                                                                                                                                 |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `data/gold/rankings/candidate_scores.csv`           | Sıra ve toplam skor                                                                                                                                    |
| `data/gold/rankings/candidate_scores_explained.csv` | Açıklamalı sürüm: `tfidf_score`, `semantic_score`, `skill_score`, `experience_score`, `final_score`, `matched_skills`, `missing_skills`, `explanation` |
| `data/gold/models/tfidf_model.pkl`                  | Eğitilmiş TF-IDF                                                                                                                                       |
| `artifacts/runs/<UTC>/manifest.json`                | Config özeti, artifact yolları, metrikler                                                                                                              |


Etiketli `ground_truth.csv` için ölçülen metrikler:

- **Precision@K** — ilk K adayda doğru aday oranı
- **Recall@K** — etiketli adayların ne kadarı ilk K’ye girdi
- **NDCG@K (graded)** — sıra duyarlı dereceli skor
- **MRR** — ilk doğru adayın mertebesi
- **MAP** — ortalama precision

İlgili modüller: `src/evaluation/metrics.py`, `src/evaluation/ranking_metrics.py`.

Çalıştırma:

```bash
python main.py --evaluate
```

---

## 8. Açıklanabilirlik

Her aday için raporda gösterilenler:

- Skor bileşenleri (4 kanal + final)
- Eşleşen beceriler
- Eksik beceriler
- Deneyim notu (yeterli / eksik / belirsiz)

Bu sayede hem işe alım uzmanı hem de teknik analiz için skor "kara kutu" değildir.

---

## 9. Sürdürülebilirlik

- Tüm önemli ayarlar `config/config.yaml` üzerinden.
- Her koşunun manifest izi `artifacts/runs/<UTC>/`.
- Birim testler: `tests/` (`pytest -q`).
- CI: `.github/workflows/ci.yml`.

---

*Belge sürümü: 2.0 — 2026-05*