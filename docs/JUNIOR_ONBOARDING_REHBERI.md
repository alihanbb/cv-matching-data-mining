# Junior Onboarding Rehberi: CV–Job Matching Data Mining Projesi

Bu doküman, projeye yeni katılan **junior seviyede ve sistemi hiç bilmeyen** bir kişinin projeyi, mimariyi, veri akışını, geliştirme sürecini ve dashboard kullanımını adım adım anlaması için hazırlanmıştır.

Doküman iki amaç taşır:

1. Junior geliştirici / analist için projeyi öğretmek.
2. Bu projede yapılan mimari seçimlerin ve geliştirme kararlarının nedenlerini açıklamak.

---

## 1. Projenin Amacı

Bu proje bir **CV–Job Description eşleştirme sistemi**dir.

Sistem temel olarak şu soruya cevap verir:

> Bu iş ilanına en uygun CV hangisi ve neden?

Bunu sadece basit metin benzerliği ile değil, birkaç farklı sinyali birleştirerek yapar:

- CV metni ile iş ilanı metni ne kadar benziyor?
- İlanda istenen beceriler CV’de var mı?
- Adayın deneyim yılı yeterli mi?
- Anlamsal olarak CV ve ilan birbirine yakın mı?
- Sonuç neden böyle çıktı, açıklanabiliyor mu?

Proje şu anda demo yapılabilir durumdadır. Streamlit dashboard çalışır, iş ilanı seçilebilir ve adaylar uygunluk skoruna göre sıralanır.

---

## 2. Neden Bu Proje Bu Şekilde Tasarlandı?

Bu proje sadece “CV parser” veya sadece “TF-IDF similarity” uygulaması olarak bırakılmadı. Çünkü gerçek bir CV eşleştirme sisteminde tek bir metin benzerliği skoru yeterli değildir.

Seçilen yaklaşım:

- **TF-IDF**: Hızlı, açıklanabilir ve akademik baseline sağlar.
- **Semantic embedding**: Aynı anlama gelen ama farklı kelimelerle yazılmış metinleri yakalar.
- **Skill matching**: İş ilanındaki teknik gereksinimlerin CV’de karşılanıp karşılanmadığını açıkça gösterir.
- **Experience matching**: Adayın deneyim yılı ile ilan beklentisini karşılaştırır.
- **Late fusion**: Farklı skorları kontrollü ve açıklanabilir şekilde birleştirir.
- **Dashboard**: Teknik olmayan kişilerin sonucu görmesini sağlar.
- **Evaluation metrikleri**: Sistemin gerçekten iyi olup olmadığını ölçer.

Bu yüzden proje bir **veri madenciliği + bilgi keşfi + açıklanabilir sıralama sistemi** olarak ele alınmıştır.

---

## 3. Genel Mimari

Proje klasörü:

```text
cv-matching-data-mining/
```

Ana yapı:

```text
cv-matching-data-mining/
├── app/
│   └── streamlit_app.py
├── config/
│   └── config.yaml
├── data/
│   ├── bronze/
│   ├── silver/
│   ├── gold/
│   └── evaluation/
├── docs/
├── src/
│   ├── ingest/
│   ├── preprocessing/
│   ├── extraction/
│   ├── features/
│   ├── scoring/
│   ├── models/
│   ├── evaluation/
│   └── pipeline/
├── tests/
├── main.py
└── pyproject.toml
```

Junior kişi önce şunu anlamalıdır:

- `main.py` sistemi başlatır.
- `src/` asıl kodun olduğu yerdir.
- `data/` veri katmanlarını tutar.
- `app/` dashboard’u tutar.
- `docs/` proje açıklamaları ve raporlarıdır.
- `tests/` sistemin bozulmadığını kontrol eder.

---

## 4. Yapılan Geliştirme İşlemleri ve Nedenleri

Bu bölüm, projeyi güçlendirmek için yapılan önemli işlemleri ve seçim nedenlerini açıklar.

### 4.1 Streamlit Dashboard Eklendi

Eklenen dosya:

```text
app/streamlit_app.py
```

Neden eklendi?

- Sadece CSV çıktısı teknik kullanıcı için yeterlidir; ancak demo için görsel arayüz gerekir.
- İş ilanı seçimi, Top-N aday filtresi, skor bileşenleri ve açıklamalar tek ekranda görülebilir.
- Projenin akademik ve ürünleşebilir yönünü daha görünür yapar.

Dashboard şu dosyadan okur:

```text
data/gold/rankings/candidate_scores_explained.csv
```

Önemli karar:

> Dashboard modeli yeniden çalıştırmaz. Sadece Gold çıktıyı okur.

Bu kararın nedeni:

- UI ile model pipeline ayrışır.
- Dashboard hızlı açılır.
- Model çıktıları tekrar üretilebilir kalır.
- Hata ayıklama kolaylaşır.

---

### 4.2 Açıklanabilir Skor Çıktısı Standartlaştırıldı

Üretilen dosya:

```text
data/gold/rankings/candidate_scores_explained.csv
```

Kolonlar:

```text
job_id
cv_id
rank_for_job
tfidf_score
semantic_score
skill_score
experience_score
final_score
matched_skills
missing_skills
explanation
```

Neden yapıldı?

- Sadece `score` kolonu tek başına yeterli değildir.
- Kullanıcı “bu aday neden birinci geldi?” sorusuna cevap bekler.
- `matched_skills` ve `missing_skills` açıklanabilirliği artırır.
- `explanation` deneyim uyumunu okunur hale getirir.

---

### 4.3 Semantic Matching Eklendi

Kullanılan model:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Çalıştırma:

```powershell
python main.py --semantic
```

Neden eklendi?

- TF-IDF kelime bazlıdır; anlamı sınırlı yakalar.
- Örneğin “backend service development” ile “REST API microservice implementation” farklı kelimeler içerir ama anlam olarak yakındır.
- Semantic embedding bu tür ilişkileri daha iyi yakalar.

Neden opsiyonel flag ile eklendi?

- Model ilk seferde HuggingFace’den iner.
- CPU’da daha yavaş çalışır.
- Hızlı testlerde gerekmez.
- Bu nedenle `--no-semantic` ile kapatılabilir.

---

### 4.4 Skill Matching Eklendi

İlgili dosyalar:

```text
src/extraction/skill_extractor.py
src/scoring/fusion.py
```

Mantık:

```text
skill_score = ortak_beceriler / tüm_beceriler
```

Neden eklendi?

- İş ilanlarında beceri gereksinimleri kritik öneme sahiptir.
- CV metni genel olarak benzer olabilir ama gerekli beceriler eksik olabilir.
- Skill matching, HR ve teknik ekip için kolay açıklanabilir bir sinyal üretir.

Örnek:

```text
Job skills: Python, Docker, AWS
CV skills: Python, Docker, Kubernetes

matched_skills = Python, Docker
missing_skills = AWS
```

---

### 4.5 Experience Matching Eklendi

İlgili dosyalar:

```text
src/extraction/experience_extractor.py
src/scoring/fusion.py
```

Mantık:

```text
CV deneyimi >= ilan beklentisi → 1.0
CV deneyimi < ilan beklentisi → oranlı skor
Bilgi yoksa → nötr / default yaklaşım
```

Neden eklendi?

- Bir adayın teknik becerileri uygun olsa bile deneyim yılı yetersiz olabilir.
- Özellikle senior / junior ayrımında deneyim sinyali önemli hale gelir.
- Bu skor, final score içinde destekleyici sinyal olarak kullanılır.

---

### 4.6 Bronze / Silver / Gold Veri Mimarisi Korundu ve Güçlendirildi

Neden bu mimari seçildi?

- Veri madenciliği projelerinde ham veri, temiz veri ve model çıktısı ayrılmalıdır.
- Ham dosyaları doğrudan model çıktılarıyla karıştırmak hata ayıklamayı zorlaştırır.
- Bronze / Silver / Gold ayrımı, data engineering açısından daha profesyonel ve sürdürülebilirdir.

Katmanlar:

```text
Bronze → Ham dosyalar
Silver → Temizlenmiş tablolar
Gold → Model ve skor çıktıları
```

---

### 4.7 Ingest Hatası Düzeltildi

Önceki problem:

```text
pandas.errors.EmptyDataError: No columns to parse from file
```

Sebep:

- `data/bronze/job_descriptions/` altında dosya yoktu.
- Ingest boş `cleaned_jobs.csv` yazıyordu.
- Pandas boş dosyayı okuyamıyordu.

Yapılan düzeltme:

- `src/ingest/build_processed.py` artık boş durumda bile başlıklı CSV üretir.
- `src/pipeline/orchestrator.py` daha anlaşılır hata mesajı verir.
- Bronze klasörüne örnek CV ve iş ilanı dosyaları eklendi.

Neden önemli?

- Junior kullanıcı hata aldığında ne yapacağını anlayabilmeli.
- Demo sırasında sistem “sessizce” bozulmamalı.

---

### 4.8 Evaluation Metrikleri Güncellendi

Eklenen / desteklenen metrikler:

- `Precision@K`
- `Recall@K` / Top-K hit rate
- `NDCG@K`
- `MRR`
- `MAP`

Ground truth formatı:

```text
job_id,cv_id,relevance
```

Relevance:

```text
3 = çok uygun
2 = uygun
1 = zayıf uygun
0 = uygun değil
```

Neden dereceli relevance seçildi?

- CV uygunluğu çoğu zaman sadece 0/1 değildir.
- “Çok uygun”, “orta uygun”, “zayıf uygun” gibi seviyeler vardır.
- NDCG gibi sıralama metrikleri dereceli etiketi daha iyi kullanır.

---

### 4.9 Testler Genişletildi

Eklenen test alanları:

- Text cleaning
- Skill extraction
- Experience extraction
- Score fusion
- Evaluation metrics
- Schema validation

Neden?

- Junior geliştirici değişiklik yaptığında sistemi bozup bozmadığını anlayabilmeli.
- Skor mantığı küçük değişikliklerle kolay bozulabilir.
- Testler proje davranışının yaşayan dokümantasyonudur.

Çalıştırma:

```powershell
pytest -q
```

---

### 4.10 README ve Docs Güncellendi

Eklenen / güncellenen dosyalar:

```text
README.md
docs/RAPOR.md
docs/PIPELINE_DIAGRAM.md
docs/KVKK_VE_GUVENLIK.md
docs/PROJE_KAVRAMSAL_REHBER.md
docs/CALISTIRMA_ORTAMI_VE_GELISTIRME_YONETIMI.md
```

Neden?

- Proje sadece koddan oluşmaz.
- Akademik ve profesyonel görünüm için amaç, yöntem, veri mimarisi, güvenlik ve değerlendirme açıkça anlatılmalıdır.
- Yeni gelen kişinin sistemi anlaması için dokümantasyon şarttır.

---

## 5. Veri Mimarisi: Bronze / Silver / Gold

Bu projede veri üç ana katmanda yönetilir.

### 5.1 Bronze: Ham Veri

Konum:

```text
data/bronze/cvs/
data/bronze/job_descriptions/
```

Burada ham dosyalar durur:

- `.txt`
- `.md`
- `.pdf`
- `.docx`

Örnek:

```text
data/bronze/cvs/cv_004_devops.txt
data/bronze/job_descriptions/job_003_devops_sre.txt
```

Kural:

> Bronze veri değiştirilmez. Kaynak neyse o haliyle saklanır.

---

### 5.2 Silver: Temizlenmiş Veri

Konum:

```text
data/silver/
```

Burada pipeline’ın okuyacağı temiz tablolar oluşur:

```text
cleaned_cvs.csv
cleaned_jobs.csv
unified_resumes.jsonl
```

Bu dosyalar şu komutla üretilir:

```powershell
python main.py --ingest
```

---

### 5.3 Gold: Model ve Sonuç Katmanı

Konum:

```text
data/gold/
```

Önemli çıktılar:

```text
data/gold/rankings/candidate_scores.csv
data/gold/rankings/candidate_scores_explained.csv
data/gold/models/tfidf_model.pkl
```

Dashboard özellikle şu dosyadan okur:

```text
data/gold/rankings/candidate_scores_explained.csv
```

Bu dosyada her aday için skorlar ve açıklama vardır.

---

## 6. Pipeline Süreci

Junior kişi şu akışı ezberlemelidir:

```text
Bronze dosyalar
    ↓
Ingest
    ↓
Silver CSV
    ↓
Text preprocessing
    ↓
Feature extraction
    ↓
Scoring
    ↓
Ranking
    ↓
Gold output
    ↓
Streamlit dashboard
```

Komut karşılığı:

```powershell
python main.py --ingest
python main.py --semantic --evaluate
streamlit run app/streamlit_app.py
```

---

## 7. Skor Mantığı

Sistem dört skor üretir.

### 7.1 TF-IDF Score

Dosya:

```text
src/features/tfidf_vectorizer.py
```

Amaç:

- CV ve iş ilanı metinlerini kelime bazlı vektörlere çevirir.
- Cosine similarity ile benzerlik ölçer.

Bu baseline modeldir.

---

### 7.2 Semantic Score

Dosya:

```text
src/features/semantic_encoder.py
```

Model:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Amaç:

- Aynı anlama gelen ama farklı kelimelerle yazılmış ifadeleri yakalamak.

Örnek:

```text
"backend service development"
"REST API microservice implementation"
```

Kelime olarak farklı olabilir ama anlam olarak yakındır.

---

### 7.3 Skill Score

Dosyalar:

```text
src/extraction/skill_extractor.py
src/scoring/fusion.py
```

Amaç:

- CV’deki becerileri çıkarır.
- İş ilanındaki becerileri çıkarır.
- Jaccard similarity hesaplar.

Formül:

```text
skill_score = ortak_beceriler / tüm_beceriler
```

---

### 7.4 Experience Score

Dosyalar:

```text
src/extraction/experience_extractor.py
src/scoring/fusion.py
```

Amaç:

- CV’den deneyim yılını bulmak.
- İş ilanından minimum deneyim beklentisini bulmak.
- Aday yeterliyse yüksek skor vermek.

Basit mantık:

```text
CV deneyimi >= ilan beklentisi → 1.0
CV deneyimi < ilan beklentisi → oranlı skor
Bilgi yoksa → default / nötr skor
```

---

## 8. Final Score

Dosyalar:

```text
config/config.yaml
src/scoring/fusion.py
```

Formül:

```text
final_score =
0.35 * tfidf_score +
0.35 * semantic_score +
0.20 * skill_score +
0.10 * experience_score
```

Semantic kapalı çalıştırılırsa sistem kalan skorları yeniden normalize eder.

Örneğin:

```powershell
python main.py --no-semantic
```

Bu durumda semantic skoru kullanılmaz.

---

## 9. Açıklanabilir Çıktı

Dosya:

```text
data/gold/rankings/candidate_scores_explained.csv
```

Önemli kolonlar:

```text
job_id
cv_id
rank_for_job
tfidf_score
semantic_score
skill_score
experience_score
final_score
matched_skills
missing_skills
explanation
```

Junior kişi özellikle bu dosyayı incelemelidir. Çünkü dashboard’un gösterdiği veri buradan gelir.

---

## 10. Dashboard

Dosya:

```text
app/streamlit_app.py
```

Çalıştırma:

```powershell
streamlit run app/streamlit_app.py
```

Dashboard şunları gösterir:

- İş ilanı seçimi
- Top-N aday filtresi
- Aday sıralaması
- Final score
- Skor bileşenleri
- Matched skills
- Missing skills
- Explanation
- CV metni

Junior kişi dashboard kodunu okurken şunu bilmelidir:

> Dashboard model çalıştırmaz. Sadece `candidate_scores_explained.csv` dosyasını okur.

Yani dashboard’da veri güncellenmiyorsa önce pipeline tekrar çalıştırılmalıdır:

```powershell
python main.py --semantic --evaluate
```

---

## 11. Evaluation / Metrikler

Dosya:

```text
data/evaluation/ground_truth.csv
```

Format:

```text
job_id,cv_id,relevance
job_003_devops_sre,cv_004_devops,3
```

Relevance anlamı:

```text
3 = çok uygun
2 = uygun
1 = zayıf uygun
0 = uygun değil
```

Metrikler:

- `Precision@K`
- `Recall@K` / Top-K hit rate
- `NDCG@K`
- `MRR`
- `MAP`

Kodlar:

```text
src/evaluation/metrics.py
src/evaluation/ranking_metrics.py
```

Junior kişi şunu anlamalıdır:

> Modelin iyi olup olmadığını sadece göze bakarak değil, ground truth ve metriklerle ölçeriz.

---

## 12. En Önemli Kod Dosyaları

Junior için inceleme sırası şöyle olmalıdır:

1. `README.md`
2. `docs/RAPOR.md`
3. `docs/PIPELINE_DIAGRAM.md`
4. `main.py`
5. `src/pipeline/orchestrator.py`
6. `src/ingest/build_processed.py`
7. `src/preprocessing/cleaner.py`
8. `src/extraction/skill_extractor.py`
9. `src/extraction/experience_extractor.py`
10. `src/scoring/fusion.py`
11. `src/models/matcher.py`
12. `app/streamlit_app.py`
13. `tests/`

Neden bu sıra?

- Önce proje amacı ve genel akış anlaşılır.
- Sonra giriş noktası ve pipeline okunur.
- Daha sonra skor bileşenleri detaylandırılır.
- En son dashboard ve testler incelenir.

---

## 13. Junior İçin İlk 7 Günlük Yol Haritası

### Gün 1: Projeyi Çalıştır

Komutlar:

```powershell
cd C:\Users\ACER\Desktop\cv_analysis\cv-matching-data-mining
.\.venv\Scripts\activate
pytest -q
python main.py --no-semantic --evaluate
streamlit run app/streamlit_app.py
```

Hedef:

- Testler geçsin.
- Pipeline çalışsın.
- Dashboard açılsın.

---

### Gün 2: Veri Katmanlarını İncele

Bakılacak yerler:

```text
data/bronze/
data/silver/
data/gold/
data/evaluation/
```

Hedef:

- Bronze nedir?
- Silver nedir?
- Gold nedir?
- Dashboard hangi dosyayı okuyor?

---

### Gün 3: Pipeline Kodunu Oku

Öncelik:

```text
main.py
src/pipeline/orchestrator.py
```

Hedef:

- `--ingest`
- `--semantic`
- `--evaluate`

flag’lerinin ne yaptığını anlamak.

---

### Gün 4: Skor Bileşenlerini Anla

Oku:

```text
src/features/tfidf_vectorizer.py
src/features/semantic_encoder.py
src/extraction/skill_extractor.py
src/extraction/experience_extractor.py
src/scoring/fusion.py
```

Hedef:

- TF-IDF nedir?
- Semantic score nedir?
- Skill score nasıl hesaplanıyor?
- Experience score nasıl hesaplanıyor?
- Final score nasıl oluşuyor?

---

### Gün 5: Evaluation Mantığını Öğren

Oku:

```text
data/evaluation/ground_truth.csv
src/evaluation/metrics.py
src/evaluation/ranking_metrics.py
```

Hedef:

- `relevance` ne demek?
- `NDCG@K` neden önemli?
- `MRR` neyi ölçüyor?

---

### Gün 6: Dashboard’u İncele

Oku:

```text
app/streamlit_app.py
```

Hedef:

- Dashboard hangi CSV’yi okuyor?
- Job seçimi nasıl çalışıyor?
- Top-N filtresi nasıl çalışıyor?
- Aday açıklaması nasıl gösteriliyor?

---

### Gün 7: Küçük Geliştirme Yap

Junior’a verilecek ilk basit görevler:

- Dashboard’da kolon adlarını Türkçeleştirmek.
- Skorları yüzde formatında göstermek.
- `matched_skills` rozetlerini daha okunaklı yapmak.
- Yeni bir örnek CV ekleyip pipeline sonucunu gözlemlemek.
- Yeni bir skill alias eklemek (`postgresql -> sql` gibi).
- Test yazmak.

---

## 14. Junior İçin Yapılmaması Gerekenler

Şunları yapmamalı:

- `data/gold/` dosyalarını elle düzenlememeli.
- `candidate_scores_explained.csv` elle değiştirilmemeli.
- Önce pipeline çalıştırmadan dashboard’da sonuç beklememeli.
- Büyük model eklemeden önce test ve metriklere bakmadan karar vermemeli.
- Gerçek CV verisini anonimleştirmeden kullanmamalı.
- Notebook’ta yazdığı kodu kalıcı pipeline gibi bırakmamalı.

---

## 15. Geliştirme Süreci Nasıl Yönetilmeli?

Her geliştirme şu sırayla yapılmalı:

```text
1. Problem tanımı
2. Hipotez
3. Küçük kod değişikliği
4. Test
5. Pipeline çalıştırma
6. Metrik kontrolü
7. Dashboard kontrolü
8. Dokümantasyon güncelleme
```

Örnek:

```text
Problem:
Data Scientist ilanında bazı ML adayları düşük skor alıyor.

Hipotez:
Skill alias listesinde "ml" -> "machine learning" eklenirse skor iyileşir.

Kod:
src/extraction/skill_extractor.py

Test:
tests/test_skill_extractor.py

Doğrulama:
python main.py --semantic --evaluate
streamlit run app/streamlit_app.py
```

---

## 16. Junior İçin Mini Sözlük

- **CV**: Aday özgeçmişi
- **Job Description**: İş ilanı
- **Ingest**: Ham dosyayı okunabilir tabloya dönüştürme
- **Bronze**: Ham veri
- **Silver**: Temizlenmiş veri
- **Gold**: Model çıktısı / skor çıktısı
- **TF-IDF**: Kelime önemine göre metin vektörleştirme
- **Cosine Similarity**: İki vektörün benzerliği
- **SBERT / Semantic**: Anlamsal metin benzerliği
- **Skill Matching**: Ortak becerileri ölçme
- **Experience Matching**: Deneyim yılı uyumu
- **Late Fusion**: Farklı skorları ağırlıklı birleştirme
- **Ground Truth**: Doğru kabul edilen etiketli veri
- **NDCG**: Doğru adaylar üst sıralarda mı, onu ölçer
- **MRR**: İlk doğru aday kaçıncı sırada, onu ölçer
- **MAP**: Genel sıralama kalitesini ölçer

---

## 17. Junior’a Verilecek İlk Görev Listesi

Başlangıç için uygun görevler:

1. `README.md` ve `docs/RAPOR.md` oku.
2. `python main.py --no-semantic --evaluate` çalıştır.
3. `candidate_scores_explained.csv` dosyasını incele.
4. Streamlit dashboard’u aç.
5. `job_003_devops_sre` için neden `cv_004_devops` birinci geliyor, açıklamasını yaz.
6. Yeni bir örnek CV ekle.
7. `python main.py --ingest --semantic --evaluate` çalıştır.
8. Yeni CV dashboard’da görünüyor mu kontrol et.
9. `tests/test_skill_extractor.py` içine yeni bir test ekle.
10. Değişiklik sonrası `pytest -q` çalıştır.

---

## 18. Senior Bakışla Özet

Bu projede junior’ın öğrenmesi gereken en kritik fikir şudur:

> Bu sistem sadece “metin benzerliği” yapmıyor; CV ve iş ilanını çok kanallı olarak analiz ediyor, skorları açıklanabilir şekilde birleştiriyor ve sonuçları metriklerle ölçüyor.

Önce veri akışını, sonra skor bileşenlerini, sonra evaluation mantığını, en son dashboard’u öğrenmelidir. Bu sırayı takip ederse projeye sağlıklı katkı verebilir.

---

## 19. Bu Dokümanın Kullanımı

Bu belge junior onboarding için kullanılmalıdır.

Önerilen kullanım:

1. Junior kişiye bu belge okutulur.
2. İlk gün sadece çalıştırma ve dashboard gösterimi yapılır.
3. İkinci gün veri katmanları ve `candidate_scores_explained.csv` incelenir.
4. Üçüncü gün `orchestrator.py` üzerinden pipeline anlatılır.
5. İlk görev olarak küçük bir dashboard iyileştirmesi veya skill alias testi verilir.

Bu yaklaşım, kişinin projeye doğrudan karmaşık model kodundan değil, sistem akışı ve veri mantığından başlamasını sağlar.

---

*Son güncelleme: 2026-05-05*
