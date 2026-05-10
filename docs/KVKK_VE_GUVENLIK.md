# KVKK ve güvenlik

## Dış veri importu

Dış CV veri kaynakları kişisel veri içerebilir. Import edilen Bronze `raw_text` alanları **skorlama öncesi** PII anonymizer’dan geçirilmelidir (`privacy.anonymize`).

Maskelenecek bilgiler (örnek): e-posta, telefon, URL, adres; isim bilgisi mümkünse anonimleştirilmelidir. Skorlamada kişisel kimlik bilgileri kullanılmamalıdır.

## Kişisel veri

CV ve iş ilanı dosyaları aday ve işverenlere ait kişisel veri içerebilir. Bu repo, **skorlama öncesi** opsiyonel PII maskeleme (`privacy.anonymize`) ile e-posta, URL ve telefon benzeri kalıpları `src/preprocessing/pii.py` üzerinden `[REDACTED]` ile gizlemeyi destekler.

## Saklama ve erişim

- Bronze katmanı yalnızca yetkili ortamlarda tutulmalı; paylaşım öncesi anonimleştirme veya rıza süreçleri ayrıca yönetilmelidir.
- Üretilen `candidate_scores_explained.csv` dosyası eşleşme açıklamaları içerir; iç paylaşımda RBAC uygulayın.

## Günlük ve manifest

`artifacts/runs/` altındaki manifestler yapılandırma ve metrik içerir; hammadde metin içermez.

## Öneriler

- Üretim ortamında ayrı bir veri sınıflandırma politikası ve saklama süresi tanımlayın.
- Dashboard’u yalnızca VPN veya iç ağ üzerinden yayınlayın.
