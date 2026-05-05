# Geliştirme ve sürdürülebilirlik rehberi

Bu belge, projeyi **uzun vadede** sağlıklı tutmak için süreç önerileri, sürümleme ve operasyonel alışkanlikleri tanımlar.

---

## 1. Yerel geliştirme

```bash
cd cv-matching-data-mining
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -e ".[dev]"
# İsteğe bağlı:
pip install -e ".[semantic]"
pytest -q
python main.py --no-semantic
```

**Dal stratejisi (öneri):** `main` / `develop` korunur; özellikler `feature/<kısa-açıklama>` dallarında gelir; PR ile birleştirilir.

**Commit mesajları:** Ne değişti ve neden (bir cümle yeterli); config veya veri yolu değiştiyse belgede veya PR açıklamasında belirtin.

---

## 2. Bağımlılık yönetimi

- **Kaynak:** `pyproject.toml` (tek doğruluk kaynağı).
- **İsteğe bağlı:** Ortam kilidi (`uv lock`, `poetry lock`) — yol haritası Faz 1.5.
- Yeni kütüphane eklerken: üretim bağımlılığı `dependencies`, geliştirme aracı `optional-dependencies.dev`, ağır opsiyoneller `semantic`.

---

## 3. Test ve regresyon

- Yeni **skorlama / fusion / şema** davranışı → ilgili `tests/` altında birim testi.
- Pipeline davranışı değişince: `python main.py --no-semantic` ile duman testi.
- Ground truth güncellenirse: metriklerin beklenen yönünü PR’da not edin (ör. “NDCG@3 artmalı”).

---

## 4. Yapılandırma ve deneyler

- Üretim veya paylaşılan deneyler: `config.yaml` değişikliği **ayrı commit** veya açık PR başlığı.
- `experiment.write_manifest: true` iken her koşuda `artifacts/runs/` altında iz kaydı oluşur; önemli koşuların manifest path’i not alınabilir.
- Ham metin **manifest’e yazılmaz**; yalnızca yol ve hash — KVKK ile uyumlu tasarım.

---

## 5. Sürümleme (semver önerisi)

| Bileşen | Örnek | Ne zaman artar? |
|---------|--------|------------------|
| MAJOR | 1.0.0 | Geriye dönük uyumsuz API/şema |
| MINOR | 0.3.0 | Yeni özellik, geriye uyumlu |
| PATCH | 0.2.1 | Hata düzeltmesi, dokümantasyon |

`pyproject.toml` içindeki `version` alanı release ile güncellenir; git etiketi (`v0.2.0`) isteğe bağlıdır.

---

## 6. Veri yaşam döngüsü

- **Bronze:** Mümkünse değiştirilmez; yeni veri yeni dosya veya tarihli alt klasör.
- **Silver:** Ingest çıktısı; kaynak dosya listesi veya hash manifest’e bağlanabilir (gelecek iyileştirme).
- **Gold:** Model ve sıralama çıktıları yeniden üretilebilir; kritik üretim artifact’ları için registry (yol haritası Faz 3).

---

## 7. Güvenlik ve uyumluluk

- Özet ve kontrol listesi: [KVKK_VE_GUVENLIK.md](KVKK_VE_GUVENLIK.md).
- Loglarda tam CV metninden kaçının.
- Erişim: `data/` ve `artifacts/` için ortam bazlı izin modeli.

---

## 8. Dokümantasyon güncelliği

Aşağıdakiler anlamlı mimari değişiklikte güncellenmelidir:

| Dosya | Ne zaman? |
|--------|-----------|
| [MEVCUT_DURUM_VE_MIMARI.md](MEVCUT_DURUM_VE_MIMARI.md) | Modül, veri katmanı veya akış değişince |
| [YOL_HARITASI.md](YOL_HARITASI.md) | Öncelik veya faz hedefi revizyonu |
| `README.md` | Kullanıcı komutları veya kurulum değişince |
| `data/README.md` | Dizin sözleşmesi değişince |

---

## 9. Hızlı kontrol listesi (PR öncesi)

- [ ] `pytest` geçiyor
- [ ] `python main.py --no-semantic` hatasız
- [ ] `config/config.yaml` örnekleri ve varsayılanlar tutarlı
- [ ] Gerekirse dokümantasyon veya yol haritası güncellendi

---

*Belge sürümü: 1.0 — 2026-05*
