# Yol haritası (ürün ve teknik)

Bu belge, projenin **uzun ömürlü** ve **evrilebilir** kalması için önerilen geliştirme planını fazlara ayırır. Öncelikler iş değeri, risk ve bağımlılık düşünülerek sıralanmıştır; her faz sonunda ölçülebilir bir çıktı hedeflenir.

---

## Vizyon

**Kısa:** Güvenilir, açıklanabilir offline CV–ilan eşleştirmesi ve raporlanabilir kalite metrikleri.

**Uzun:** Ölçeklenebilir veri hattı, öğrenilmiş veya hibrit sıralama, kurumsal güvenlik ve (isteğe bağlı) servis API’si ile entegre aday önerisi.

---

## Faz 0 — Sağlam temel (sürekli)

**Hedef:** Mevcut mimariyi bozmadan kaliteyi korumak.


| Madde                   | Çıktı / başarı ölçütü                                |
| ----------------------- | ---------------------------------------------------- |
| CI yeşil                | Her PR’da `pytest` + `main.py --no-semantic`         |
| Config tek doğruluk     | Yeni davranışlar `config.yaml` + küçük kod           |
| Ground truth genişletme | En az bir “regresyon” iş–aday seti; metrikler trendi |
| Dokümantasyon           | Mimari / yol haritası PR’larla güncellenir           |


**Süre:** sürekli.

---

## Faz 1 — Veri ve ölçüm (0–3 ay)

**Hedef:** Kararlar veriye dayansın; Silver/Gold üretimi tekrarlanabilir olsun.


| #   | Madde                 | Notlar                                                          |
| --- | --------------------- | --------------------------------------------------------------- |
| 1.1 | Etiketleme protokolü  | Çoklu ilan, çok aday; `relevant` tanımı yazılı                  |
| 1.2 | Veri sürümleme        | Ham/silver snapshot adı veya DVC benzeri iz; manifest’e bağlama |
| 1.3 | Segment metrikleri    | Dil, pozisyon ailesi, kaynak bazlı NDCG/MRR                     |
| 1.4 | Ingest sağlamlaştırma | Bozuk PDF, boş çıktı, dil tespiti bayrakları                    |
| 1.5 | Lock dosyası          | `uv.lock` / `poetry.lock` ile ortam sabitleme                   |


**Çıktı:** Genişletilmiş `ground_truth`, metrik panosu (en az tablo/rapor), daha az “elle müdahale”.

---

## Faz 2 — Model ve sıralama (3–9 ay)

**Hedef:** Semantik ve yapısal sinyalleri güçlendirmek; manuel ağırlık bağımlılığını azaltmak.


| #   | Madde                    | Notlar                                               |
| --- | ------------------------ | ---------------------------------------------------- |
| 2.1 | BM25 veya hybrid lexical | İlan odaklı lexical iyileştirme                      |
| 2.2 | Cross-encoder / rerank   | İlk aday kümesini ikinci aşamada yeniden sıralama    |
| 2.3 | Öğrenilmiş fusion        | Validation set ile ağırlık veya hafif LTR modeli     |
| 2.4 | Beceri grafiği / synonym | Ontoloji veya eş anlamlı sözlük; Jaccard iyileştirme |
| 2.5 | Deneyim kuralları        | İlan bandı, “zorunlu beceri” hard/soft constraint    |


**Çıktı:** Offline NDCG/MRR’de ölçülebilir iyileşme; üretim öncesi karşılaştırma raporu.

---

## Faz 3 — Ürünleştirme ve uyumluluk (9–18 ay)

**Hedef:** Güvenli, izlenebilir operasyon.


| #   | Madde                       | Notlar                                               |
| --- | --------------------------- | ---------------------------------------------------- |
| 3.1 | Servis API (isteğe bağlı)   | Senkron skorlama; şema sürümü                        |
| 3.2 | Auth, rate limit, audit log | KVKK dokümanı ile uyum                               |
| 3.3 | Model registry              | Hangi artifact’ın üretimde olduğu kayıt altında      |
| 3.4 | Drift izleme                | Skor dağılımı, gecikme, hata oranı alarmları         |
| 3.5 | İnsan geri bildirim döngüsü | Tıklama / mülakat daveti → kalibrasyon (online eval) |


**Çıktı:** Üretim kontrol listesi karşılanır; olay müdahalesi runbook’u.

---

## Riskler ve bağımlılıklar


| Risk                     | Azaltma                                                   |
| ------------------------ | --------------------------------------------------------- |
| Etiket yetersizliği      | Faz 1’i önce kilitlemek                                   |
| Embedding model maliyeti | CPU-only ve küçük model seçenekleri; `--no-semantic` yolu |
| Veri kalitesi            | Bronze kalite kontrolü ve Silver şema validasyonu         |
| Regülasyon               | `KVKK_VE_GUVENLIK.md` güncel tutulur                      |


---

## Nasıl güncellenir?

Bu yol haritası **üç ayda bir** veya önemli milestone sonrası gözden geçirilir. Tamamlanan maddeler işaretlenir; yeni işler Faz 0’a veya uygun faza eklenir.

---

*Belge sürümü: 1.0 — 2026-05*