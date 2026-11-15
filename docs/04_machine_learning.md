# Modelleme veri hattı

Bu aşama yalnızca model girdilerini, hedefi ve kronolojik veri bölümlerini hazırlar. Model eğitimi ve Streamlit uygulaması bu kapsamda değildir.

## Tahmin anı ve hedef

Her satırın tahmin anı **t**'dir. Model, t anında erişilebilen son ölçümü ve geçmişi kullanarak **t+10 dakika** zamanındaki `Appliances` değerini (Wh) tahmin eder.

- `origin_time`: tahminin yapıldığı an, t
- `target_time`: tahmin edilen ölçümün zamanı, t+10 dakika
- `target_appliances_wh`: t+10 dakikada ölçülen hedef değer
- `current__...`: t anında ölçülen diğer sensör/hava durumu sütunları
- `appliances_lag_0`: t anındaki son Appliances ölçümü
- `appliances_lag_1`: t-10 dakika ölçümü; diğer lag'ler geriye gider
- `appliances_rolling_mean_N`: t dahil t ve öncesindeki son N ölçümün ortalaması

Hareketli ortalama pencereleri sağa hizalıdır (`center=False`); gelecekteki satırları kullanmaz. Son satırın t+10 hedefi bulunmadığından ve geçmiş penceresi tamamlanmayan ilk satırlar bulunduğundan bu satırlar eğitim tablosundan çıkarılır.

## İki satırlık örnek

| Ölçüm zamanı | Appliances (Wh) | T1 |
|---|---:|---:|
| 10:00 (t) | 30 | 21.0 |
| 10:10 (t+10) | 42 | 21.5 |

10:00 anında oluşturulan örnek:

| Alan | Kullanılan değer | Açıklama |
|---|---:|---|
| `appliances_lag_0` | 30 Wh | t anındaki son Appliances ölçümü |
| `current__T1` | 21.0 | t anındaki sıcaklık |
| `target_time` | 10:10 | Hedef zamanı; model girdisi değildir |
| `target_appliances_wh` | 42 Wh | Öğrenilecek hedef; model girdisi değildir |

**Kullanılmaz:** 10:10 satırındaki T1=21.5 ve diğer gelecekteki sensör sütunları. 42 Wh yalnızca hedef etikettir; özellik üretiminde kullanılmaz.

## Veri bölme

`chronological_split`, örnekleri `target_time` alanına göre eskiden yeniye sıralar ve varsayılan olarak %70 eğitim, %15 doğrulama, %15 test bölümü oluşturur. Karıştırma yapılmaz. En yeni hedef zamanları test bölümünde kalır.

## Kullanım

```python
from energy_forecast.data_analysis.data_loader import load_energy_data
from energy_forecast.machine_learning.features import (
    build_supervised_dataset, chronological_split, get_feature_columns,
)

raw_data = load_energy_data()
supervised = build_supervised_dataset(raw_data)
train, validation, test = chronological_split(supervised)
feature_columns = get_feature_columns(supervised)

X_train, y_train = train[feature_columns], train["target_appliances_wh"]
X_validation, y_validation = validation[feature_columns], validation["target_appliances_wh"]
X_test, y_test = test[feature_columns], test["target_appliances_wh"]
```

`target_time` raporlama ve zaman bazlı değerlendirme içindir; `get_feature_columns` tarafından model girdilerinden çıkarılır.
