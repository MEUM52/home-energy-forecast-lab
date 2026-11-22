# 06 — AI Application

Bu aşama, önceden hazırlanmış bir makine öğrenmesi modeliyle **geçmiş test kayıtları üzerinde 10 dakikalık enerji tüketimi tahmini** gösteren tek ekranlı bir Streamlit demosudur.

> **Canlı izleme değildir.** Gerçek zamanlı sensör veya cihaz bağlantısı yoktur. Kayıtlar, model karşılaştırmasında ayrılmış test bölümünden gelir.

## Ekranda neler var?

- Test bölümünden tahmin kaydı seçimi
- Tahmin anı ve hedef zaman
- Wh / 10 dakika cinsinden tahmin
- Kullanıcının belirlediği tüketim eşiğine göre uyarı
- Gerçekleşen tüketim ve mutlak hata (yalnızca **Gerçekleşen değeri göster** düğmesine basıldığında)

Tahmin seçilen eşiği aşarsa **“Yüksek tüketim bekleniyor”** uyarısı gösterilir.

## 1. Ortamı kur

Python 3.10 veya üzeri önerilir. Proje ana klasöründe:

### Windows PowerShell

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Ham veri dosyasını yerleştir

UCI [Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction) veri setini indirin. CSV dosyasını aşağıdaki konuma ve dosya adına yerleştirin:

```text
data/raw/energydata_complete.csv
```

Ham veri Git'e eklenmez.

## 3. Demo dosyasını hazırla (eğitim bu adımda yapılır)

Proje ana klasöründe bir kez çalıştırın:

```bash
python 06_AI_Application/prepare_demo.py
```

Bu komut mevcut veri yükleme, özellik oluşturma ve kronolojik ayırma işlevlerini kullanır. Baseline, Linear Regression ve Random Forest modellerini eğitim bölümünde eğitir; doğrulama MAE değerine göre bir model seçer ve seçilen modeli test kayıtlarıyla birlikte şu dosyaya kaydeder:

```text
models/energy_forecast_demo.joblib
```

Test bölümü model seçimi için kullanılmaz. Hazırlama komutu yeniden çalıştırılırsa model yeniden eğitilir ve demo dosyası güncellenir.

## 4. Streamlit uygulamasını aç

```bash
streamlit run 06_AI_Application/streamlit_app.py
```

**Önemli:** Streamlit uygulaması açılışta model eğitmez. Yalnızca `models/energy_forecast_demo.joblib` dosyasını yükler ve seçilen test kaydı için tahmin üretir.

## Dosya eksikse

Uygulama demo dosyasını bulamazsa ekranda şu adımlar gösterilir:

1. UCI veri setini indirin.
2. `data/raw/energydata_complete.csv` konumuna yerleştirin.
3. `python 06_AI_Application/prepare_demo.py` komutunu çalıştırın.
4. Streamlit komutuyla uygulamayı yeniden açın.

Hazırlama komutu ham veri dosyasını bulamazsa beklenen tam dosya yolunu terminalde bildirir.

## Birimler ve yorum

- Tahmin, hedef zamana ait 10 dakikalık enerji tüketimidir.
- Tahmin ve eşik Wh / 10 dakika birimindedir.
- Tahmin eşikten **büyükse** yüksek tüketim uyarısı görünür.
- Mutlak hata: `|gerçekleşen değer − tahmin|`.

Bu arayüz bir geçmiş veri demosudur; gerçek bir evin anlık tüketimini izlemez ve operasyonel alarm sistemi olarak kullanılmamalıdır.

## Yerel dosya

Model ve demo test kayıtları makinede üretilir. `models/` klasörünün içeriği Git tarafından yok sayılır; başka bir bilgisayarda kullanmak için 2. ve 3. adımları yeniden uygulayın.
