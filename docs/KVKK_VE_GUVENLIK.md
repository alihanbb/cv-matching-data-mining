# KVKK ve güvenlik notları (özet)

Bu proje **kişisel veri** (CV, iletişim, eğitim, iş geçmişi) işlediği varsayımıyla değerlendirilmelidir.

## Veri minimizasyonu ve saklama

- Ham dosyaları yalnızca gerekli süre için tutun; amaç sona erince **silme veya anonimleştirme** politikası tanımlayın.
- `artifacts/runs/*/manifest.json` içinde **ham metin** tutmayın; yalnızca dosya yolu ve hash kullanın (mevcut uygulama buna uygundur).

## Loglama

- Üretim ortamında stdout/log dosyalarına **tam CV metni** yazmayın.
- Hata ayıklama için kısa özet veya maskelenmiş örnek kullanın.

## Erişim kontrolü

- `data/bronze`, `data/silver` ve `data/gold` içeriğini yalnızca yetkili hesaplara açın; paylaşılan dizinlerde erişim listelerini sınırlayın.

## Model önyargısı

- Skorlar yalnızca **aday önerisi** içindir; işe alım kararı insan sürecine bağlanmalıdır.
- Cinsiyet, yaş, medeni durum gibi **hassas alanları** skor üretimine doğrudan dahil etmeyin; vekil değişken riskini gözden geçirin.

## Üretim API (ileride)

- Oran sınırlama (rate limit), kimlik doğrulama ve denetim günlüğü (kim, hangi aday için sorgu yaptı) planlayın.

Bu metin hukuki danışmanlık yerine geçmez; kurumsal kullanımda uyum ekibiyle doğrulayın.