# 04 — Machine Learning

Bu aşamada üç yöntem aynı hedef zamanları üzerinde karşılaştırılır:

1. **Son ölçüm (Baseline):** t anındaki son Appliances ölçümünü t+10 tahmini kabul eder.
2. **Linear Regression:** Eğitim bölümündeki örneklerle doğrusal regresyon öğrenir.
3. **Random Forest:** Eğitim bölümündeki örneklerle 200 ağaçlı regresyon modeli öğrenir.

Hedef, t+10 dakikadaki `Appliances` tüketimidir ve Wh cinsindendir.

## Veri sızıntısını önleme ve adil karşılaştırma

- Özellikler yalnızca t anında veya daha önce erişilebilen ölçümlerden üretilir.
- `target_time` ve `target_appliances_wh` model girdisi değildir.
- Üç yöntem aynı train, validation ve test satırlarını kullanır.
- Öğrenen iki model yalnızca eğitim bölümünde fit edilir.
- Baseline, her örnekte `appliances_lag_0` değerini tahmin olarak döndürür.
- Model seçimi doğrulama MAE değerine göre yapılır.
- Test bölümü seçim için kullanılmaz; seçim tamamlandıktan sonra üç yöntemin nihai performansını raporlamak için kullanılır.

## Metrikler

- **MAE (Wh):** Mutlak tahmin hatalarının ortalaması. Küçük olması daha iyidir.
- **RMSE (Wh):** Hataların karelerinin ortalamasının karekökü. Büyük hataları daha fazla cezalandırır. Küçük olması daha iyidir.

Her iki metrik doğrudan Wh cinsinden raporlanır. Gerçek veri üzerinde çalıştırılmamış sonuç/rakam verilmez.

## Çalıştırma

CSV dosyası `data/raw/energydata_complete.csv` konumunda olmalıdır.

Windows PowerShell (proje kök dizininde, sanal ortam açıkken):

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m energy_forecast.machine_learning.model_comparison
```

macOS/Linux:

```bash
export PYTHONPATH="$PWD/src"
python -m energy_forecast.machine_learning.model_comparison
```

Komut önce doğrulama tablosunu ve doğrulama MAE değerine göre seçilen modeli, ardından üç yöntemin test MAE/RMSE değerlerini yazdırır.

## Sonuçları yorumlama

Doğrulama tablosunda Baseline ilk sıradaysa, bu veri bölümünde daha karmaşık modeller son ölçüm tahminini geçememiştir; basit yöntem açıkça en iyi model olarak raporlanır.

Test tablosu üç yöntemin son performansını gösterir. Testte baseline daha iyi görünse bile doğrulama aşamasında yapılan seçim değiştirilmez. Test, model seçmek veya hiperparametre ayarlamak için kullanılmamalıdır.

Komut çalıştırılmadan önce MAE/RMSE değerleri bilinemez; örnek veya uydurma performans rakamları kullanılmamalıdır.
