# CV–İş İlanı Eşleştirme Sistemi — Teknik Değerlendirme Raporu

**Hedef kurum:** Tesla, Inc. (mühendislik, veri ve İK paydaşları)  
**Çözüm:** `cv-matching-data-mining` — çok kanallı, açıklanabilir aday–pozisyon eşleştirme hattı  
**Rapor sürümü:** 1.0  
**Tarih:** 9 Mayıs 2026  
**Kod tabanı sürümü:** `pyproject.toml` → 0.3.0  

---

## 1. Yönetici özeti

Bu rapor, Tesla’nın aday havuzunu iş ilanlarıyla **ölçülebilir, denetlenebilir ve çok sinyalli** biçimde sıralamak için tasarlanmış bir **batch (çevrimdışı) analitik motorunu** anlatır. Sistem yalnızca “anahtar kelime eşleşmesi” değildir; **lexical (TF‑IDF, BM25)**, **semantik (sentence-transformers)** ve **yapılandırılmış beceri + deneyim** kanalları tek bir skorda birleştirilir; çıktıda kanal bazlı skorlar ve metinsel gerekçeler üretilir.

**Tesla için pratik değer:** Yüksek hacimli mühendislik ve operasyon pozisyonlarında ön eleme maliyetini düşürme, nötr ve tekrarlanabilir bir ön sıralama, mülakat öncesi “hangi yetkinlikler eksik?” sorusuna veri odaklı yanıt.

**Dürüstlük çerçevesi (satış ve güven):** Aşağıdaki **sayısal başarım metrikleri**, depodaki **küçük pilot ground truth** (3 ilan, 12 etiketli aday–ilan çifti) üzerindedir. Tesla üretim ortamına geçiş için **Tesla verisi ve etiketli değerlendirme seti** ile yeniden ölçüm ve hukuki/etik inceleme zorunludur. Bu rapor mimariyi ve mevcut ölçümleri şeffaf biçimde sunar.

---

## 2. Kapsam ve sınırlar

| Dahil | Dahil değil (yol haritası) |
|--------|----------------------------|
| Ham CV/ilan dosyalarından metin çıkarma (PDF, DOCX, TXT, MD) | Gerçek zamanlı SaaS ATS yerine geçme |
| Silver CSV üretimi, çok kanallı skorlama, top‑K sıralama | Varsayılan REST API / SSO (CLI + Streamlit + CSV) |
| PII maskeleme (e‑posta, URL, telefon, TR adres benzeri) — skorlamadan önce | Tam KVKK/GDPR uyumluluk beyanı (kurum politikası + DPIA gerekir) |
| Offline metrikler (Precision/Recall/NDCG@K, MRR, MAP) | Tesla ağında doğrulanmış latency SLO’ları |
| Ağırlık optimizasyonu, öğrenilebilir füzyon, cross‑encoder rerank (opsiyonel) | Otomatik işe alım kararı |

---

## 3. Mimari genel bakış

### 3.1 Veri katmanları (medallion benzeri)

| Katman | Konum (örnek) | İçerik |
|--------|----------------|--------|
| Bronze | `data/bronze/cvs`, `data/bronze/job_descriptions` | Ham dosyalar |
| Silver | `data/silver/cleaned_cvs.csv`, `cleaned_jobs.csv` | `id` + tam metin |
| Silver+ | `data/silver/unified_resumes.jsonl` (opsiyonel) | Bölümlenmiş CV, beceri listesi, kalite skoru |
| Gold | `data/gold/rankings/*.csv`, `data/gold/models/` | Sıralama çıktıları, TF‑IDF artefaktı |
| Artefakt | `artifacts/runs/<UTC>/manifest.json` | Config özeti, hash’ler, metrikler |

### 3.2 İşlem akışı (özet)

1. **`python main.py --ingest`** — Bronze → Silver CSV (`build_processed.py`, `text_extract.py`). İsteğe bağlı JSONL korpus birleştirme (`ingest.cv_corpus_jsonl` ile `config.yaml`).
2. **`python main.py`** — `build_matching_matrices` ile tüm kanallar; `fuse_scores` ile geç füzyon; `rank_candidates_for_jobs` ile ilan başına top‑K; `enrich_detailed` ile açıklama metinleri.
3. **Değerlendirme** — `ground_truth.csv` varsa NDCG, MAP, MRR vb.
4. **Dashboard** — `streamlit run app/streamlit_app.py` (açıklamalı CSV okur).

---

## 4. Sayısal envanter (bu repodaki ölçülen değerler)

Aşağıdaki rakamlar **bu çalışma kopyasındaki dosyalardan** türetilmiştir; üretim iddiası değildir.

### 4.1 Gümüş (Silver) eşleştirme girdisi

| Öğe | Değer | Kaynak |
|-----|--------|--------|
| İşlenmiş aday CV kaydı | **4.005** | `data/silver/cleaned_cvs.csv` (CSV parser ile kayıt sayımı; dosya çok satırlı tırnaklı alanlar içerir) |
| İşlenmiş ilan kaydı | **3** | `data/silver/cleaned_jobs.csv` |
| Teorik tam çift sayısı (kartesian) | **4.005 × 3 = 12.015** | Skorlama matrisi bu boyutta üretilir |
| Sıralama çıktısı (top‑K=10, 3 ilan) | **30** veri satırı | `candidate_scores_explained.csv` (~31 satır başlık dahil) |

### 4.2 Birleşik CV korpusu istatistikleri (JSONL)

`data/silver/unified_resumes.jsonl.stats.json`:

| Metrik | Değer |
|--------|--------|
| Toplam kayıt | **9.484** |
| Başarılı (ok) | **9.263** |
| Boş/hata | **221** |
| Örnek kaynak dağılımı | `train_json_ner`: 5.960; `resume_corpus_csv`: 2.484; diğer kaynaklar dokümante edilmiş |

Bu katman, özellikle **NER/eğitim verisi** ve geniş profil analizi senaryoları için referans korpusudur; ticari teklif kapsamında **lisans ve kullanım hakları** ayrıca doğrulanmalıdır.

### 4.3 Beceri sözlüğü

| Metrik | Değer |
|--------|--------|
| Kanonik beceri girişi | **yaklaşık 48** skill tanımı | `config/skills.yaml` (kategoriler + alias’lar) |

Tesla için: Üretimde **iç yetkinlik çerçevesi** (ör. gömülü yazılım, batarya, üretim OT, Giga IT) ile lexicon genişletmesi beklenir.

### 4.4 Ground truth (etiketli değerlendirme seti)

| Metrik | Değer |
|--------|--------|
| Etiketli satır sayısı | **12** | `data/evaluation/ground_truth.csv` |
| Kapsanan ilan sayısı | **3** | Her ilan için 4 aday derecesi (0–3 ölçeği) |

**Yorum:** Her (ilan, aday) çifti için **dereceli relevans** (0–3) desteklenir; NDCG bu dereceleri kullanır. Pilot set **istatistiksel güç açısından küçüktür**; Tesla ile sözleşme öncesi **daha büyük, çift kör veya çok değerlendiricili** ground truth önerilir.

---

## 5. Ölçülen başarım (pilot koşu)

### 5.1 Referans koşu: `artifacts/runs/20260508T204457Z/manifest.json`

Bu koşu notları: **`dense_enabled: false`** (sembolik/dense kanal kapalı), **`bm25_enabled: true`**.

| Metrik | Değer |
|--------|--------|
| Top‑K isabet oranı @1 / @3 / @5 | **1.000** / **1.000** / **1.000** |
| Precision @1 / @3 / @5 | **1.000** / **0.556** / **0.400** |
| Recall @1 / @3 / @5 | **0.389** / **0.667** / **0.778** |
| NDCG @1 / @3 / @5 | **1.000** / **0.922** / **0.937** |
| MRR | **1.000** |
| MAP | **0.713** |

**Tanım özeti (kod ile uyumlu):**

- **Top‑K isabet:** Her ilan için top‑K’da en az bir “relevant” (relevance ≥ 1) çift var mı — ilan sayısına oran.
- **Precision@K:** İlan başına top‑K içindeki relevant oranının **ortalaması**.
- **Recall@K:** İlan başına, ground truth’taki relevant adayların ne kadarının top‑K’da olduğunun **ortalaması**.
- **NDCG@K:** Dereceli etiketlerle sıra kalitesi — ilan başına ortalama.
- **MRR / MAP:** İlk relevant sırası ve ortalama precision (ilan başına ortalama).

### 5.2 Karşılaştırmalı pilot koşu (referans)

`artifacts/runs/20260506T072909Z/manifest.json` — öğrenilmiş ağırlık notu ile; NDCG@5 **~0.484**, MAP **~0.309**. Bu, **aynı küçük pilot üzerinde farklı konfigürasyonların sonuçlarını ne kadar değiştirebildiğini** göstermek için eklenmiştir; hangisinin Tesla verisinde üstün olacağı **yeniden ölçülmelidir**.

### 5.3 Tesla mühendisliğine net mesaj

Pilot için **MRR = 1.0** ve **yüksek NDCG@5**, “sistem bu üç ilanda ilgili adayı genelde üst sıralara koydu” anlamına gelir; **genelleme iddiası** ancak daha büyük etiket seti ve saha A/B testi ile verilebilir.

---

## 6. Teknik derinlik: arka planda ne yapılıyor?

### 6.1 Gizlilik ve ön işleme

- **`privacy.anonymize`** açıksa, skorlamadan önce `src/preprocessing/pii.py` e‑posta, URL, telefon ve Türkçe adres benzeri kalıpları maskelemeye çalışır. Amaç: skorların iletişim bilgisine **sızarak şişmesini** azaltmak.
- **`TextCleaner`:** stopword, lemmatization, dil ayarı (`config.yaml`).

### 6.2 Kanallar (`src/pipeline/matching_inputs.py`)

1. **Beceri çıkarımı** — YAML lexicon + alias eşlemesi.  
2. **İlan gereksinimleri** — must / nice ayrımı, kapsam matrisi.  
3. **Deneyim** — regex tabanlı yıl sinyalleri; aday maks. yıl vs ilan min. yıl.  
4. **TF‑IDF + kosinüs** — lexical benzerlik.  
5. **Dense embedding** — varsayılan `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (opsiyonel bağımlılık); model yoksa kanal kapanır.  
6. **BM25** — `rank-bm25`; retrieval benzeri lexical sinyal.

### 6.3 Füzyon

- **`fuse_scores`:** İlan (sütun) bazında min–max normalizasyonu, sonra ağırlıklı toplam — ölçek farkını giderir.  
- **Fusion V1 / V2:** V2, BM25 ağırlığını içerir (`config.yaml` içindeki `fusion` / `fusion_v2`).

### 6.4 Açıklanabilirlik

- `matcher.enrich_detailed` ve `scoring/explain`: must/nice örtüşmesi, eksik kritik beceriler, deneyim notu, **`explanation`** ve **`suggested_improvements`** metinleri.

### 6.5 İkinci aşama (opsiyonel)

- **`python main.py --rerank`:** Cross‑encoder ile top‑N çiftlerin yeniden puanlanması ve baz skor ile harmanlanması (`src/models/cross_encoder_rerank.py`).

### 6.6 Operasyonel güvenilirlik

- **`PipelineConfig` (Pydantic)** ile yapılandırma doğrulaması (`main.py`).  
- **CI:** `.github/workflows/ci.yml` — `pytest`, ardından `python main.py --no-semantic` duman testi.  
- **Manifest:** Her koşuda config hash ve girdi dosyası hash’leri (`src/utils/experiment.py`).

---

## 7. Tesla’ya özgü konumlandırma önerisi

- **Pozisyon çeşitliliği:** Tesla; yazılım, donanım, veri, üretim, güvenlik ve saha operasyonları için farklı dil ve jargon kullanır. Çok dilli embedding ve **Tesla’ya özel lexicon** (ör. AUTOSAR, ISO 26262, ECU, BMS, robotics stack) füzyon kalitesini belirler.
- **Ölçek:** Tam matris O(N_cv × N_job) — on binlerce ilan ve milyonlarca profil için **iki aşamalı retrieval** (BM25/ANN ön seçim + cross‑encoder rerank) tipik enterprise mimarisidir.
- **Denetim:** Mevcut `manifest.json` ve `score_audit` sütunları **model yönetimi ve regresyon testi** için başlangıç noktasıdır; Tesla MLOps ile genişletilebilir.

---

## 8. İnsan Kaynakları ve etik kullanım

- Çıktılar **karar destek** amaçlıdır; otomatik red veya teklif üretimi insan sürecinin yerine konmamalıdır.  
- **Bias:** Lexicon ve eğitilmiş modeller tarihsel önyargı taşıyabilir; düzenli denetim ve şeffaflık (kanal uzayı ve açıklama metni) riski azaltmaya yardımcı olur.  
- **Erişim:** Açıklamalı CSV içerikleri hassas olabilir; RBAC ve VPN önerisi `docs/KVKK_VE_GUVENLIK.md` ile uyumludur.

---

## 9. Çalıştırma referansı

```bash
# Geliştirme kurulumu
pip install -e ".[dev,bm25,semantic,dashboard]"

# Bronze → Silver
python main.py --ingest

# Tam pipeline (semantic açık varsayılan; CI’da --no-semantic)
python main.py --bm25

# Değerlendirme (ground truth gerekli)
python main.py --evaluate

# Dashboard
streamlit run app/streamlit_app.py
```

---

## 10. Sonuç

Bu çözüm, Tesla’nın mühendislik ve İK ekiplerine **çok kanallı, açıklanabilir ve yapılandırması denetlenebilir** bir aday sıralama hattı sunar. **Mevcut sayısal metrikler**, depodaki **küçük ama dereceli pilot** üzerinde güçlü sıralama sinyali göstermektedir; **Tesla verisi üzerinde doğrulama** ve **hukuki çerçeve** tamamlandığında üretim teklifi sağlamlaşır.

---

## Ek A: Dosya ve modül haritası (özet)

| Bileşen | Dosya |
|---------|--------|
| Giriş noktası | `main.py` |
| Orkestrasyon | `src/pipeline/orchestrator.py` |
| Özellik matrisleri | `src/pipeline/matching_inputs.py` |
| Füzyon | `src/scoring/fusion.py` |
| Sıralama ve zenginleştirme | `src/models/matcher.py` |
| PII | `src/preprocessing/pii.py` |
| Değerlendirme | `src/evaluation/metrics.py`, `ranking_metrics.py` |
| Panel | `app/streamlit_app.py` |
| Konfigürasyon | `config/config.yaml`, `src/config/schema.py` |

---

## Ek B: Workspace içindeki ayrı veri varlığı

Kök `cv_analysis/README.md` altında **5.960+ özette NER eğitim örneği** açıklanmaktadır; bu, eşleştirme motorundan **bağımsız bir veri ürünüdür** ve lisans kontrolü ile birlikte sunulmalıdır.

---

## Ek C: Uygulama ekran görüntüleri (Streamlit dashboard)

Aşağıdaki görseller, `streamlit run app/streamlit_app.py` ile açılan **CV–Job Matching Dashboard** arayüzünden alınmıştır (`data/gold/rankings/candidate_scores_explained.csv` mevcutken). Dosyalar `docs/screenshots/` altındadır; PDF veya sunuma aktarırken aynı görselleri yüksek çözünürlükte kullanabilirsiniz.

**Candidate Ranking** — İlan seçimi, top‑N, çok kanallı skor tablosu ve ilan metni özeti:

![Candidate Ranking sekmesi](screenshots/01-candidate-ranking.jpeg)

**CV Profile Analysis** — Silver CSV üzerinden tekil CV incelemesi:

![CV Profile Analysis sekmesi](screenshots/02-cv-profile.jpeg)

**Requirement Coverage** — Must / nice örtüşmesi ve eşleşen zorunlu beceriler:

![Requirement Coverage sekmesi](screenshots/03-requirement-coverage.jpeg)

**Evaluation Metrics** — Ground truth sonrası metrik CSV’si için yer tutucu (üretimde `python main.py --export-eval-csv` ile doldurulur):

![Evaluation Metrics sekmesi](screenshots/04-evaluation-metrics.jpeg)

**Model Comparison (NDCG)** — Model karşılaştırma CSV’si oluşturulmadığında bilgilendirme mesajı (aynı komutla üretilir):

![Model Comparison sekmesi](screenshots/05-model-comparison.jpeg)

**Score Debug** — Skor denetimi (`final_score_raw` vs `score_check`) ve semantik kanal sıfır oranı (bu koşumda dense kanal kapalı olduğu için **%100** sıfır görülebilir):

![Score Debug sekmesi](screenshots/06-score-debug.jpeg)
