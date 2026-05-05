# CV Analizi Projesi: Kavramsal Rehber ve Veri Bilimi Bakışı

Bu doküman, **cv-matching-data-mining** projesini senior data analyst / data scientist bakışıyla açıklar. Amaç; projenin neyi çözmeye çalıştığını, veri setlerinin nasıl konumlandığını, bilgi keşfi ve veri madenciliği sürecinin hangi aşamalardan oluştuğunu ve mevcut teknik yapının hangi hedefe hizmet ettiğini anlaşılır hale getirmektir.

---

## 1. Projenin Ana Amacı

Bu projenin temel amacı, **CV metinleri ile iş ilanları arasında ölçülebilir, açıklanabilir ve tekrar üretilebilir bir uyum skoru** üretmektir.

Sistem sadece “metin benzerliği” yapmak için değil, CV analizi kapsamında şu işleri yapacak şekilde tasarlanmalıdır:

- CV ve iş ilanı metinlerinden anlamlı bilgi çıkarmak.
- Farklı veri setlerini tek bir standart veri altyapısında toplamak.
- Metin madenciliği ve bilgi keşfi yöntemleri ile aday–iş uyumunu analiz etmek.
- Adayları ilanlara göre sıralamak.
- Skorların neden yüksek veya düşük olduğunu açıklayabilmek.
- Sonuçları offline metriklerle ölçmek ve zaman içinde iyileştirmek.

Bu nedenle proje bir **CV parser** projesinden daha geniştir. Parser yalnızca bir bileşendir. Asıl sistem; veri toplama, veri kalite kontrolü, bilgi çıkarımı, özellik üretimi, modelleme, sıralama, değerlendirme ve raporlama adımlarından oluşan uçtan uca bir veri bilimi hattıdır.

---

## 2. Problem Tanımı

İnsan kaynakları süreçlerinde bir iş ilanına çok sayıda CV gelebilir. Bu CV’leri manuel incelemek zaman alır ve tutarlılık sorunu yaratır. Bu proje, şu soruya veri odaklı cevap üretmeyi hedefler:

> “Bu iş ilanı için hangi CV’ler daha uygundur ve neden?”

Bu soruyu cevaplamak için sistem üç ana sinyal kullanır:

1. **Metinsel benzerlik:** CV metni ile iş ilanı metni ne kadar benziyor?
2. **Yapılandırılmış beceri uyumu:** İlanda istenen beceriler CV’de var mı?
3. **Deneyim ve bağlam uyumu:** CV’deki unvan, deneyim yılı, alan ve sektör ilanla örtüşüyor mu?

Gelişmiş aşamada dördüncü sinyal olarak **semantik/anlamsal benzerlik** kullanılır. Bu, aynı beceri veya deneyim farklı kelimelerle ifade edildiğinde daha doğru eşleştirme sağlar.

---

## 3. Projede Kullanılan Veri Setleri

Projede farklı kaynaklardan gelen CV verileri vardır. Bunların hepsi aynı formatta değildir; bu nedenle ilk önemli hedef **veri setlerini tek bir canonical formatta birleştirmektir**.

### 3.1 NER Eğitim Veri Seti

Kök dizindeki `train.json`, CV metinleri ve bu metinler üzerindeki entity annotation bilgilerini içerir.

Örnek entity tipleri:

- `SKILL`
- `DESIGNATION`
- `LOCATION`
- `EXPERIENCE`
- `PERSON`
- `EDUCATION`
- `EMAIL`
- `COMPANY`
- `CERTIFICATION`

Bu veri seti doğrudan aday sıralama modeli değildir. Asıl değeri, CV içinden beceri, unvan, eğitim, deneyim gibi alanları öğrenilmiş bir NER modeliyle çıkarabilme potansiyelidir.

Mevcut istatistik:

- `train.json`: 5.960 kayıt
- `sample.json`: 5 kayıt
- `Entity Recognition in Resumes.json`: 220 kayıt

### 3.2 Kategori Bazlı PDF CV Veri Seti

`data/data/<KATEGORI>/*.pdf` altında 24 kategoriye ayrılmış PDF CV’ler bulunur.

Örnek kategoriler:

- `INFORMATION-TECHNOLOGY`
- `ENGINEERING`
- `FINANCE`
- `HR`
- `SALES`
- `ACCOUNTANT`
- `TEACHER`

Mevcut durum:

- 24 kategori
- 2.484 PDF CV

Bu veri seti, pozisyon/kategori bazlı değerlendirme için değerlidir. Örneğin bir “Software Engineer” ilanında IT veya Engineering kategorisindeki CV’lerin üst sıralarda çıkması beklenebilir.

### 3.3 CSV Tabanlı Yapılandırılmış CV Veri Setleri

Projede iki önemli CSV kaynağı vardır:

- `resume_dataset_2.csv`
- `Resume/Resume.csv`

`resume_dataset_2.csv`, isim, e-posta, telefon, üniversite, deneyim yılı, rol, beceriler ve CV metni gibi alanlar içerir.

`Resume/Resume.csv`, kategori ve resume metni içeren daha geniş bir kaynaktır. Bu dosyadaki 2.484 kayıt, PDF veri setinin metne dönüştürülmüş haliyle örtüşüyor olabilir. Bu yüzden PDF ve CSV birlikte kullanılırken duplicate kontrolü önemlidir.

---

## 4. Tek Formatta Birleşik Veri Altyapısı

Proje için doğru yaklaşım, tüm kaynakları doğrudan tek bir CSV’ye sıkıştırmak değil; hepsini ortak bir **canonical JSONL** formatına dönüştürmektir.

Bu amaçla eklenen script:

```bash
python -m src.ingest.unify_datasets \
  --source-root "C:\Users\ACER\Desktop\cv_analysis" \
  --output data/silver/unified_resumes.jsonl
```

Üretilen dosya:

```text
data/silver/unified_resumes.jsonl
```

Mevcut birleşik veri durumu:

- Toplam kayıt: 9.484
- Başarılı / boş olmayan kayıt: 9.263
- Boş veya hatalı metin: 221
- Dosya boyutu: yaklaşık 57.96 MB

Kaynak dağılımı:


| Kaynak                    | Kayıt |
| ------------------------- | ----- |
| `train_json_ner`          | 5.960 |
| `sample_json_ner`         | 5     |
| `entity_recognition_json` | 220   |
| `resume_dataset_2_csv`    | 815   |
| `resume_corpus_csv`       | 2.484 |


### 4.1 Canonical Kayıt Şeması

Birleşik JSONL dosyasında her satır tek bir CV kaydını temsil eder.

Ana alanlar:

- `record_id`: Tekil kayıt kimliği
- `source`: Kaynağın adı
- `source_file`: Kaynak dosya
- `source_row`: Kaynak satır numarası
- `document_type`: Şu an `resume`
- `category`: Rol/kategori bilgisi
- `language`: Dil bilgisi
- `text`: CV metni
- `text_length`: Metin uzunluğu
- `labels`: Annotation, beceri veya kategori etiketleri
- `metadata`: Kaynağa özgü ek bilgiler
- `extraction_status`: `ok`, `empty_text`, `extract_error`
- `error`: Hata varsa açıklaması

Bu yapı, farklı veri setlerini tek altyapıda taşımaya uygundur.

---

## 5. Bilgi Keşfi (KDD) Süreci

Bu proje, klasik **KDD - Knowledge Discovery in Databases** yaklaşımına göre ele alınmalıdır.

### 5.1 Veri Seçimi

Amaç, hangi verilerin hangi problem için kullanılacağını belirlemektir.

Bu projede:

- NER verisi bilgi çıkarımı için kullanılır.
- Kategori bazlı CV verisi sınıflandırma ve sıralama değerlendirmesi için kullanılır.
- CSV veri setleri hızlı baseline ve model geliştirme için kullanılır.
- İş ilanı verileri aday–iş eşleştirmesinin hedef tarafını oluşturur.

### 5.2 Veri Temizleme

Amaç, bozuk, eksik, tekrar eden veya kullanılamayan kayıtları tespit etmektir.

Kontrol edilmesi gerekenler:

- Boş CV metinleri
- Bozuk PDF parse sonuçları
- Duplicate kayıtlar
- Çok kısa veya anlamsız metinler
- Hatalı annotation aralıkları
- Kategori tutarsızlıkları
- Kişisel veri hassasiyeti

### 5.3 Veri Dönüşümü

Amaç, farklı formatlardaki verileri ortak modele dönüştürmektir.

Bu projede dönüşüm:

- PDF/CSV/JSON kaynaklarını unified JSONL formatına aktarma
- Metin temizleme
- Beceri, unvan, eğitim, deneyim gibi alanları çıkarma
- Kategori ve kaynak bilgisini metadata olarak saklama
- Model girdisi için Silver/Gold katmanlarını oluşturma

### 5.4 Veri Madenciliği

Amaç, veriden örüntü ve karar sinyalleri çıkarmaktır.

Kullanılan veya planlanan yöntemler:

- TF-IDF + cosine similarity
- Skill Jaccard similarity
- Deneyim yılı uyumu
- Dense embedding / semantic similarity
- NER tabanlı bilgi çıkarımı
- Hibrit scoring
- İleri aşamada learning-to-rank veya reranking

### 5.5 Yorumlama ve Değerlendirme

Amaç, üretilen sonuçların anlamlı, güvenilir ve açıklanabilir olup olmadığını ölçmektir.

Ölçümler:

- Precision@K
- Recall@K
- NDCG@K
- MRR
- MAP
- Parser doluluk oranı
- Kaynak bazlı hata oranı
- Kategori bazlı performans

---

## 6. Mevcut Proje Mimarisi

Projenin ana klasörü:

```text
cv-matching-data-mining/
```

Önemli dizinler:

```text
cv-matching-data-mining/
├── main.py
├── config/
├── data/
│   ├── bronze/
│   ├── silver/
│   ├── gold/
│   └── evaluation/
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
├── docs/
└── artifacts/
```

### 6.1 Bronze Katmanı

Ham verinin saklandığı katmandır.

Örnek:

- PDF CV dosyaları
- DOCX/TXT CV dosyaları
- Ham iş ilanı dosyaları
- Orijinal veri kaynakları

Bu katmanda veri değiştirilmez. Amaç, kaynağı izlenebilir tutmaktır.

### 6.2 Silver Katmanı

Temizlenmiş ve normalize edilmiş veri katmanıdır.

Örnek çıktılar:

- `cleaned_cvs.csv`
- `cleaned_jobs.csv`
- `unified_resumes.jsonl`

Bu katman modelleme için doğrudan kullanılabilir hale gelmiş metin ve metadata içerir.

### 6.3 Gold Katmanı

Model çıktılarının ve iş değeri üreten sonuçların bulunduğu katmandır.

Örnek:

- `candidate_scores.csv`
- `candidate_scores_explained.csv`
- `tfidf_model.pkl`

Gold katmanı, karar destek sistemine veya raporlama sürecine beslenebilecek çıktıları içerir.

---

## 7. Modelleme Yaklaşımı

Proje şu anda çok kanallı bir skorlama yaklaşımı kullanır.

### 7.1 TF-IDF Kanalı

CV ve iş ilanı metinleri sayısal vektörlere dönüştürülür. Sonra cosine similarity ile benzerlik hesaplanır.

Avantajları:

- Hızlıdır.
- Açıklanabilirdir.
- Baseline için uygundur.

Sınırları:

- Eş anlamlı kelimeleri her zaman yakalayamaz.
- “Machine learning” ile “ML model development” gibi ifadeleri tam anlayamayabilir.

### 7.2 Skill Kanalı

Beceri listeleri çıkarılır ve CV ile ilan arasındaki ortak beceriler ölçülür.

Örnek:

- İlan: Python, SQL, Docker
- CV: Python, PostgreSQL, Docker, FastAPI
- Ortak beceriler: Python, Docker

Bu kanal, CV–ilan uyumunda çok güçlü ve açıklanabilir bir sinyal üretir.

### 7.3 Deneyim Kanalı

CV’deki deneyim yılı veya deneyim ifadesi ile ilandaki beklenti karşılaştırılır.

Örnek:

- İlan: 5+ yıl deneyim
- CV: 6 yıl backend geliştirme

Bu durumda deneyim uyumu yüksek kabul edilir.

### 7.4 Semantic Embedding Kanalı

Opsiyonel olarak sentence-transformers ile dense embedding kullanılır.

Avantajı:

- Aynı anlamı farklı kelimelerle ifade eden metinleri daha iyi eşleştirir.

Örnek:

- “REST API development”
- “backend service integration”

Bu iki ifade lexical olarak birebir aynı olmasa da semantik olarak yakın olabilir.

### 7.5 Late Fusion

Her kanal ayrı skor üretir. Daha sonra bu skorlar normalize edilir ve ağırlıklı şekilde birleştirilir.

Örnek ağırlık mantığı:

- TF-IDF: metinsel uyum
- Dense embedding: anlamsal uyum
- Skill: beceri uyumu
- Experience: deneyim uyumu

Bu yaklaşım, tek bir modelin zayıflıklarını azaltır.

---

## 8. Değerlendirme Mantığı

Sistem sadece skor üretmemelidir; ürettiği skorun kalitesi ölçülmelidir.

Bu nedenle `data/evaluation/ground_truth.csv` gibi bir dosya önemlidir.

Beklenen şema:

```text
cv_id,job_id,relevant
```

Burada:

- `relevant = 1`: CV ilgili iş için uygun
- `relevant = 0`: CV ilgili iş için uygun değil

Ölçülen metrikler:

- **Precision@K:** İlk K adayın kaç tanesi gerçekten uygun?
- **NDCG@K:** Doğru adaylar üst sıralarda mı?
- **MRR:** İlk doğru aday kaçıncı sırada geliyor?
- **MAP:** Birden fazla doğru aday varsa genel sıralama kalitesi nasıl?

Bu metrikler olmadan modelin gerçekten iyi olup olmadığını söylemek mümkün değildir.

---

## 9. Mevcut Durum

Şu an proje şu yeteneklere sahiptir:

- Bronze/Silver/Gold veri katmanı yapısı var.
- CV ve iş ilanı eşleştirme pipeline’ı çalışıyor.
- TF-IDF tabanlı baseline var.
- Skill ve deneyim gibi yapılandırılmış sinyaller kullanılıyor.
- Açıklanabilir skor çıktısı üretiliyor.
- Test paketi mevcut ve son kontrolde testler başarılı.
- Birleşik JSONL veri seti üretildi.
- PDF dahil etme pilotu başarıyla denendi.

Son doğrulama:

```text
pytest -q
7 passed
```

Pipeline hızlı çalışma:

```bash
python main.py --no-semantic
```

Bu komut, semantic embedding kanalını kapatıp hızlı ve tekrar üretilebilir baseline çalıştırır.

---

## 10. Kritik Riskler ve Veri Bilimi Açısından Dikkat Noktaları

### 10.1 Duplicate Kayıt Riski

`Resume/Resume.csv` ile PDF veri seti aynı kaynağın farklı temsilleri olabilir. Bu nedenle PDF’leri canonical dataset’e eklemeden önce duplicate kontrolü yapılmalıdır.

Öneri:

- `text_hash`
- normalize edilmiş metin benzerliği
- source + external id kontrolü

### 10.2 Etiket Kalitesi

CV–iş eşleştirme için doğru ground truth yoksa model performansı yanıltıcı olabilir.

Öneri:

- Küçük ama kaliteli bir insan etiketli değerlendirme seti hazırlanmalı.
- Her iş ilanı için en az 20–50 aday etiketlenmeli.
- Etiketleme kriterleri yazılı hale getirilmeli.

### 10.3 Parser Kalitesi

PDF parse çıktıları hatalı olabilir. Bazı PDF’lerde metin boş, karışık veya eksik çıkabilir.

Öneri:

- Parser başarı oranı kaynak/kategori bazlı ölçülmeli.
- Bozuk PDF’ler ayrı hata sınıfı olarak tutulmalı.
- Gerekirse OCR veya alternatif parser karşılaştırması yapılmalı.

### 10.4 Bias ve KVKK

CV verileri kişisel veri içerir. Ayrıca modelin belirli okul, lokasyon, cinsiyet veya isim sinyallerine gereğinden fazla ağırlık vermesi risktir.

Öneri:

- Kişisel veriler sadece gerekli ise kullanılmalı.
- Açıklanabilirlik çıktılarında hassas alanlar dikkatli yönetilmeli.
- KVKK ve veri saklama politikası uygulanmalı.

---

## 11. Önerilen Aşamalar

### Aşama 1 — Veri Kalitesi ve Tekilleştirme

Hedef:

- Birleşik JSONL dosyasının kalite raporunu çıkarmak.
- Duplicate ve boş metinleri tespit etmek.
- Kategori ve kaynak dağılımını netleştirmek.

Çıktılar:

- `unified_resumes_quality_report.md`
- Duplicate aday listesi
- Kaynak bazlı doluluk oranı

### Aşama 2 — Parser ve Bilgi Çıkarımı

Hedef:

- CV metinlerinden beceri, unvan, eğitim, deneyim alanlarını daha güvenilir çıkarmak.
- MiningResume regex/LLM parser çıktıları ile NER veri setini karşılaştırmak.

Çıktılar:

- Parser kalite metriği
- Alan doluluk raporu
- NER eğitim/deney planı

### Aşama 3 — Baseline Eşleştirme

Hedef:

- TF-IDF + skill + experience fusion ile ilk güvenilir skorlayıcıyı sabitlemek.

Çıktılar:

- `candidate_scores.csv`
- `candidate_scores_explained.csv`
- Baseline metrik raporu

### Aşama 4 — Semantik ve Hibrit Model

Hedef:

- Çok dilli embedding kanalını üretim benzeri deneylere dahil etmek.
- TF-IDF ve embedding skorlarını karşılaştırmak.

Çıktılar:

- Semantic vs non-semantic karşılaştırma
- Kategori bazlı performans raporu

### Aşama 5 — Öğrenilmiş Sıralama ve Geri Bildirim

Hedef:

- İnsan etiketleri veya işe alım geri bildirimleri ile modeli kalibre etmek.

Çıktılar:

- Genişletilmiş ground truth
- Learning-to-rank deneyi
- İnsan geri bildirim döngüsü

---

## 12. Kısa Vadeli Öncelik Listesi

En doğru sonraki işler:

1. `unified_resumes.jsonl` için kalite raporu üretmek.
2. Duplicate detection yapmak.
3. PDF ve CSV kaynakları arasında örtüşme kontrolü yapmak.
4. Ground truth etiketleme protokolünü yazmak.
5. İlk 3–5 iş ilanı için küçük ama kaliteli değerlendirme seti oluşturmak.
6. Parser alan doluluk oranlarını ölçmek.
7. Baseline skorları kategori bazında analiz etmek.

---

## 13. Sonuç

Bu proje, klasik bir CV parser projesinden daha kapsamlıdır. Doğru konumlandırma şu şekildedir:

> CV verilerini tek veri altyapısında toplayan, metin madenciliği ve bilgi çıkarımı yapan, iş ilanı–aday uyumunu çok kanallı skorlayan ve sonuçlarını ölçülebilir metriklerle değerlendiren bir veri bilimi sistemi.

Şu an projenin temeli kurulmuştur. En kritik sonraki adım, model geliştirmeden önce **veri kalitesi, tekilleştirme ve ground truth** sürecini sağlamlaştırmaktır. Bu yapılmadan daha karmaşık modeller eklemek kısa vadede iyi görünse de uzun vadede güvenilir sonuç üretmez.

---

*Son güncelleme: 2026-05-05*