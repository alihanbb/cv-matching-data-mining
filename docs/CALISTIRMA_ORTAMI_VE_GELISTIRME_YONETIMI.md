# Çalıştırma Ortamı ve Geliştirme Yönetimi Rehberi

Bu doküman, **cv-matching-data-mining** projesinin nerede, hangi ortamda ve nasıl çalıştırılacağını; iyileştirme ve geliştirme süreçlerinin senior data analyst / data scientist bakışıyla nasıl yönetilmesi gerektiğini açıklar.

Amaç, projeyi sadece “bir Python scripti” olarak değil, veri kalitesi, deney yönetimi, model geliştirme, test, dokümantasyon ve ölçüm disiplinleri olan sürdürülebilir bir veri bilimi sistemi olarak ele almaktır.

---

## 1. Projeyi Nerede Çalıştıracağım?

Bu proje şu an için en uygun şekilde **yerel geliştirme ortamında** çalıştırılır.

Mevcut ana proje dizini:

```text
C:\Users\ACER\Desktop\cv_analysis\cv-matching-data-mining
```

Çalıştırma için önerilen ana ortam:

```text
Windows + Cursor IDE + PowerShell + Python virtual environment
```

Bu ortam, geliştirme, veri keşfi, pipeline denemesi, test çalıştırma ve dokümantasyon için yeterlidir.

---

## 2. Önerilen Ortam Tipleri

Projeyi tek bir ortam gibi düşünmemek gerekir. Veri bilimi projelerinde farklı amaçlar için farklı çalışma modları olmalıdır.

### 2.1 Lokal Geliştirme Ortamı

Kullanım amacı:

- Kod geliştirme
- Küçük veriyle hızlı deneme
- Unit test çalıştırma
- Pipeline duman testi
- Dokümantasyon güncelleme

Önerilen araçlar:

- Cursor IDE
- PowerShell
- Python 3.10+
- `.venv`
- `pytest`

Bu proje için günlük geliştirme ortamı budur.

### 2.2 Deney / Analiz Ortamı

Kullanım amacı:

- Veri keşfi
- Kategori dağılımı analizi
- Duplicate kontrolü
- Model karşılaştırması
- Metrik analizi

Önerilen araçlar:

- Jupyter Notebook veya `.py` analiz scriptleri
- `notebooks/` klasörü
- `artifacts/runs/` deney çıktıları
- `data/silver/` ve `data/gold/` çıktıları

Not: Notebook hızlı keşif için iyidir; kalıcı pipeline mantığı notebook içinde bırakılmamalıdır. Çalışan analiz kodu zamanla `src/` altına taşınmalıdır.

### 2.3 Batch / Pipeline Ortamı

Kullanım amacı:

- Tam veri setiyle tekrarlanabilir pipeline çalıştırma
- Bronze → Silver → Gold üretimi
- Skor dosyaları ve manifest üretimi

Örnek komut:

```powershell
cd C:\Users\ACER\Desktop\cv_analysis\cv-matching-data-mining
python main.py --no-semantic
```

Semantic model kuruluysa:

```powershell
python main.py
```

Tam ingest ile:

```powershell
python main.py --ingest --no-semantic
```

### 2.4 Üretim Benzeri Ortam

Henüz şart değil, ancak ileride gerekir.

Kullanım amacı:

- Zamanlanmış batch işler
- API üzerinden skor üretme
- İnsan kaynakları sistemine entegrasyon
- Model/versiyon izleme

Önerilen yapı:

- Linux sunucu veya container
- Sabit Python ortamı
- Versiyonlanmış config
- Loglama
- Artifact saklama
- KVKK uyumlu erişim kontrolü

Bu aşama için proje henüz hazır olmak zorunda değildir. Öncelik, önce offline pipeline’ın sağlamlaşmasıdır.

---

## 3. İlk Kurulum

PowerShell üzerinde:

```powershell
cd C:\Users\ACER\Desktop\cv_analysis\cv-matching-data-mining
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Kurulum sonrası kontrol:

```powershell
pytest -q
python main.py --no-semantic
```

Beklenen durum:

- Testler geçmeli.
- `data/gold/rankings/` altında skor dosyaları oluşmalı.
- `data/gold/models/` altında TF-IDF modeli oluşmalı.
- `artifacts/runs/` altında deney manifesti oluşmalı.

---

## 4. Semantic Model Ortamı

Semantic embedding kanalı daha iyi anlamsal eşleştirme sağlar, fakat daha ağırdır.

Kurulum:

```powershell
pip install -e ".[semantic]"
```

Çalıştırma:

```powershell
python main.py
```

Ne zaman kullanılmalı?

- Baseline oturduktan sonra
- Metrik karşılaştırması yapılacaksa
- Daha iyi anlamsal eşleştirme hedefleniyorsa
- Donanım ve indirme süresi sorun değilse

Ne zaman kullanılmamalı?

- Hızlı testte
- CI duman testinde
- Kod davranışını kontrol ederken
- Veri kalitesi henüz netleşmemişken

Bu yüzden günlük geliştirmede önerilen komut:

```powershell
python main.py --no-semantic
```

---

## 5. Proje Nasıl Çalışıyor?

Ana giriş noktası:

```text
main.py
```

Bu dosya şu parametreleri destekler:

```powershell
python main.py
python main.py --no-semantic
python main.py --ingest
python main.py --config config/config.yaml
```

Pipeline mantığı:

1. Config okunur.
2. Gerekirse ingest çalışır.
3. Silver CSV dosyaları okunur.
4. CV ve iş ilanı metinleri temizlenir.
5. TF-IDF özellikleri üretilir.
6. İsteğe bağlı semantic embedding üretilir.
7. Skill ve experience sinyalleri çıkarılır.
8. Tüm skorlar late fusion ile birleşir.
9. Adaylar iş ilanı bazında sıralanır.
10. Gold çıktıları ve deney manifesti yazılır.

---

## 6. Veri Katmanları Nerede?

Projede veri katmanları şu şekilde yönetilir:

```text
data/
├── bronze/
│   ├── cvs/
│   └── job_descriptions/
├── silver/
│   ├── cleaned_cvs.csv
│   ├── cleaned_jobs.csv
│   └── unified_resumes.jsonl
├── gold/
│   ├── models/
│   └── rankings/
└── evaluation/
    └── ground_truth.csv
```

### Bronze

Ham veri katmanıdır. Bu katmanda veri değiştirilmez.

Örnek:

- PDF CV
- DOCX CV
- TXT/MD iş ilanları

### Silver

İşlenmiş ve normalize edilmiş veri katmanıdır.

Örnek:

- `cleaned_cvs.csv`
- `cleaned_jobs.csv`
- `unified_resumes.jsonl`

### Gold

Model ve iş çıktısı katmanıdır.

Örnek:

- `candidate_scores.csv`
- `candidate_scores_explained.csv`
- `tfidf_model.pkl`

### Evaluation

Model kalitesini ölçmek için kullanılan ground truth dosyaları burada tutulur.

---

## 7. Birleşik Veri Seti Nasıl Üretilir?

Tüm CV kaynaklarını tek canonical JSONL formatına almak için:

```powershell
cd C:\Users\ACER\Desktop\cv_analysis\cv-matching-data-mining

python -m src.ingest.unify_datasets `
  --source-root "C:\Users\ACER\Desktop\cv_analysis" `
  --output data/silver/unified_resumes.jsonl
```

PDF dahil pilot çalıştırma:

```powershell
python -m src.ingest.unify_datasets `
  --source-root "C:\Users\ACER\Desktop\cv_analysis" `
  --output data/silver/unified_resumes_pdf_pilot.jsonl `
  --include-pdfs `
  --pdf-limit 20
```

Önemli karar:

PDF ve `Resume/Resume.csv` aynı kaynaktan geliyor olabilir. Bu nedenle PDF’leri tam canonical dosyaya eklemeden önce duplicate analizi yapılmalıdır.

---

## 8. Günlük Çalışma Akışı

Bir data analyst / data scientist için önerilen günlük akış:

### 8.1 Başlangıç Kontrolü

```powershell
cd C:\Users\ACER\Desktop\cv_analysis\cv-matching-data-mining
.\.venv\Scripts\activate
pytest -q
python main.py --no-semantic
```

Bu iki komut projenin temel olarak sağlam olup olmadığını gösterir.

### 8.2 Veri Değişikliği Varsa

Yeni CV veya iş ilanı eklendiyse:

```powershell
python main.py --ingest --no-semantic
```

Yeni dış veri seti eklendiyse:

```powershell
python -m src.ingest.unify_datasets `
  --source-root "C:\Users\ACER\Desktop\cv_analysis" `
  --output data/silver/unified_resumes.jsonl
```

### 8.3 Model / Skor Değişikliği Varsa

Önce hızlı test:

```powershell
pytest -q
python main.py --no-semantic
```

Sonra gerekiyorsa semantic deney:

```powershell
python main.py
```

### 8.4 Sonuç Analizi

Kontrol edilecek dosyalar:

```text
data/gold/rankings/candidate_scores.csv
data/gold/rankings/candidate_scores_explained.csv
artifacts/runs/<UTC>/manifest.json
```

Analizde sorulacak sorular:

- İlk 5 aday mantıklı mı?
- Skor bileşenleri beklenen yönde mi?
- Skill uyumu gerçekten açıklayıcı mı?
- Deneyim skoru yanlış pozitif üretiyor mu?
- Belirli kategorilerde sistem zayıf mı?

---

## 9. İyileştirme Süreçleri Nasıl Yönetilmeli?

İyileştirme süreci rastgele model denemeleriyle değil, ölçülebilir hipotezlerle yönetilmelidir.

Her iyileştirme şu formatta tanımlanmalıdır:

```text
Hipotez:
Beklenen etki:
Değiştirilecek bileşen:
Kullanılacak veri:
Başarı metriği:
Risk:
Geri alma planı:
```

Örnek:

```text
Hipotez: Skill alias sözlüğünü genişletmek, Software Engineer ilanlarında Precision@5'i artırır.
Beklenen etki: Python/Py, PostgreSQL/SQL gibi eşleşmeler daha iyi yakalanır.
Değiştirilecek bileşen: src/extraction/skill_extractor.py
Kullanılacak veri: ground_truth.csv + IT kategorisi
Başarı metriği: Precision@5 ve NDCG@5
Risk: Yanlış eş anlamlılar false positive artırabilir.
Geri alma planı: Alias değişikliğini config kontrollü yapmak.
```

---

## 10. Geliştirme Sürecinin Aşamaları

### Aşama 1 — Veri Kalitesi

Öncelik model değil, veri kalitesidir.

Yapılacaklar:

- Duplicate detection
- Boş metin oranı
- Kaynak bazlı kayıt sayısı
- Kategori dağılımı
- Çok kısa metin analizi
- PDF extraction hata oranı

Beklenen çıktı:

```text
docs/reports/unified_resumes_quality_report.md
```

### Aşama 2 — Parser / Extraction İyileştirme

Amaç, CV içinden daha doğru alanlar çıkarmaktır.

Yapılacaklar:

- Skill extraction iyileştirme
- Experience extraction iyileştirme
- NER veri setinden faydalanma
- MiningResume parser çıktılarıyla karşılaştırma

Ölçüm:

- Alan doluluk oranı
- Yanlış pozitif örnekleri
- Kaynak/kategori bazlı başarı

### Aşama 3 — Baseline Model Sabitleme

Amaç, basit ama güvenilir bir ilk model hattı oluşturmaktır.

Yapılacaklar:

- TF-IDF parametrelerini sabitleme
- Skill ve experience ağırlıklarını ölçme
- `--no-semantic` pipeline’ı referans kabul etme

Ölçüm:

- Precision@K
- NDCG@K
- MRR
- MAP

### Aşama 4 — Semantic Model Deneyleri

Amaç, dense embedding kanalının gerçekten katkı sağlayıp sağlamadığını ölçmektir.

Yapılacaklar:

- Semantic açık/kapalı karşılaştırması
- Model boyutu ve hız analizi
- Kategori bazlı performans farkı

Önemli: Semantic kanal sadece metrikte anlamlı katkı sağlıyorsa operasyonel hatta dahil edilmelidir.

### Aşama 5 — Öğrenilmiş Sıralama

Bu ileri aşamadır. Önce ground truth büyütülmelidir.

Yapılabilecekler:

- Logistic regression / LightGBM ranking
- Cross-encoder reranking
- İnsan geri bildirimi ile model kalibrasyonu

Bu aşamaya veri kalitesi ve baseline oturmadan geçilmemelidir.

---

## 11. Deney Yönetimi

Her deneyin kaydı tutulmalıdır.

Deney çıktıları:

```text
artifacts/runs/<UTC>/manifest.json
```

Manifest neden önemlidir?

- Hangi config ile çalıştığını gösterir.
- Hangi artifact’ların üretildiğini gösterir.
- Metrikleri kaydeder.
- Aynı deneyi tekrar üretmeyi kolaylaştırır.

Her önemli deneyden sonra not alınması gerekenler:

- Tarih
- Amaç
- Değişiklik
- Veri seti versiyonu
- Config farkı
- Metrik sonucu
- Kısa yorum

Örnek deney notu:

```text
Deney: skill_alias_v2
Amaç: Skill eşleşmelerini artırmak
Komut: python main.py --no-semantic
Veri: unified_resumes.jsonl v1 + mevcut cleaned_jobs.csv
Sonuç: Precision@5 değişimi ölçülecek
Karar: Metrik artarsa alias sözlüğü kalıcı hale getirilecek
```

---

## 12. Versiyonlama ve Değişiklik Yönetimi

Proje şu an git repository olarak initialize edilmemiş olabilir. Uzun vadeli çalışma için git kullanılması önerilir.

Önerilen branch yapısı:

```text
main
develop
feature/data-quality-report
feature/skill-extraction-v2
feature/unified-dataset-dedup
experiment/semantic-model-comparison
```

Commit mesajı örnekleri:

```text
Add unified dataset builder for resume sources
Improve skill extraction aliases for backend roles
Add quality report for canonical resume dataset
Document local development workflow
```

Her değişiklikte şu kontrol yapılmalıdır:

```powershell
pytest -q
python main.py --no-semantic
```

---

## 13. Config Yönetimi

Operasyonel ayarlar `config/config.yaml` içindedir.

Kod değiştirmeden yönetilmesi gerekenler:

- TF-IDF parametreleri
- Semantic model adı
- Fusion ağırlıkları
- Top-K değeri
- Girdi/çıktı yolları
- Logging seviyesi

Kural:

> Model davranışını değiştiren ayar mümkünse önce config üzerinden yönetilmelidir.

Bu yaklaşım deneyleri daha kontrollü ve tekrar üretilebilir hale getirir.

---

## 14. Test Stratejisi

Testler sadece yazılım kalitesi için değil, veri bilimi sisteminin güvenilirliği için de gereklidir.

Mevcut test komutu:

```powershell
pytest -q
```

Yeni test eklenmesi gereken durumlar:

- Yeni skor bileşeni eklendiğinde
- Fusion mantığı değiştiğinde
- Yeni schema alanı eklendiğinde
- Yeni metric eklendiğinde
- Ingest davranışı değiştiğinde
- Edge case düzeltildiğinde

Önerilen test kategorileri:

- Schema validation testleri
- Metric testleri
- Fusion testleri
- Skill extraction testleri
- Ingest / parser testleri
- End-to-end küçük pipeline testi

---

## 15. Dokümantasyon Yönetimi

Bu proje veri bilimi projesi olduğu için dokümantasyon kod kadar önemlidir.

Güncel tutulması gereken dosyalar:

```text
README.md
docs/PROJE_KAVRAMSAL_REHBER.md
docs/CALISTIRMA_ORTAMI_VE_GELISTIRME_YONETIMI.md
docs/MEVCUT_DURUM_VE_MIMARI.md
docs/YOL_HARITASI.md
docs/KVKK_VE_GUVENLIK.md
data/README.md
```

Ne zaman doküman güncellenmeli?

- Yeni veri kaynağı eklendiğinde
- Pipeline adımı değiştiğinde
- Config parametresi değiştiğinde
- Yeni metrik eklendiğinde
- Yeni model yaklaşımı eklendiğinde
- Veri saklama veya KVKK kararı değiştiğinde

---

## 16. Kişisel Veri ve Güvenlik

CV verileri kişisel veri içerir. Bu yüzden geliştirme sürecinde şu kurallar uygulanmalıdır:

- Loglara tam CV metni yazılmamalı.
- E-posta, telefon, isim gibi alanlar raporlarda gerekmedikçe gösterilmemeli.
- Public repository’ye ham CV verisi eklenmemeli.
- `data/` ve `artifacts/` klasörleri dikkatli yönetilmeli.
- Model açıklamalarında hassas alanların etkisi kontrol edilmeli.

Bu konu için ayrıca:

```text
docs/KVKK_VE_GUVENLIK.md
```

---

## 17. Senior Data Analyst / Data Scientist Bakışıyla Çalışma Prensibi

Bu projede doğru çalışma sırası şudur:

```text
1. Veri setini anla
2. Veri kalitesini ölç
3. Tek formatta güvenilir veri katmanı oluştur
4. Basit baseline kur
5. Baseline'ı metrikle ölç
6. Hata analizi yap
7. Küçük ve kontrollü iyileştirme dene
8. Metrik iyileşirse kalıcılaştır
9. Dokümante et
10. Test ve pipeline ile doğrula
```

Yanlış çalışma sırası:

```text
1. Hemen büyük model eklemek
2. Ground truth olmadan başarı iddiası yapmak
3. Duplicate veriyle metrik ölçmek
4. Parser hatalarını model problemi sanmak
5. Notebook'taki kodu üretim mantığı gibi bırakmak
```

---

## 18. Kısa Vadeli Yol Haritası

Önümüzdeki pratik adımlar:

1. `unified_resumes.jsonl` için kalite raporu üret.
2. Duplicate detection scripti ekle.
3. PDF ve CSV kaynak örtüşmesini ölç.
4. `ground_truth.csv` için etiketleme protokolü oluştur.
5. 3–5 örnek iş ilanı için elle etiketlenmiş aday seti hazırla.
6. Skill extraction hata analizini yap.
7. Baseline skorlarını kategori bazında raporla.
8. Semantic modelin gerçekten katkı sağlayıp sağlamadığını ölç.

---

## 19. Hızlı Komut Özeti

Kurulum:

```powershell
cd C:\Users\ACER\Desktop\cv_analysis\cv-matching-data-mining
python -m venv .venv
.\.venv\Scripts\activate
pip install -e ".[dev]"
```

Test:

```powershell
pytest -q
```

Hızlı pipeline:

```powershell
python main.py --no-semantic
```

Ingest + pipeline:

```powershell
python main.py --ingest --no-semantic
```

Semantic pipeline:

```powershell
pip install -e ".[semantic]"
python main.py
```

Birleşik veri üretimi:

```powershell
python -m src.ingest.unify_datasets `
  --source-root "C:\Users\ACER\Desktop\cv_analysis" `
  --output data/silver/unified_resumes.jsonl
```

---

## 20. Sonuç

Bu projeyi şu anda en doğru şekilde **yerel Python sanal ortamında, Cursor IDE ve PowerShell üzerinden** çalıştırmalısınız. Geliştirme süreci ise kod yazma merkezli değil, veri kalitesi ve deney yönetimi merkezli olmalıdır.

En önemli prensip:

> Önce veri kalitesi ve ölçüm, sonra model karmaşıklığı.

Proje olgunlaştıkça yerel batch çalışmadan üretim benzeri servis veya zamanlanmış pipeline mimarisine geçilebilir. Ancak bu geçiş için önce birleşik veri katmanı, ground truth, metrikler ve testler sağlamlaştırılmalıdır.

---

*Son güncelleme: 2026-05-05*
