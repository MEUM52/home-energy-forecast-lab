# Home Energy Forecast Lab

Python ve makine öğrenmesine giriş yapan öğrenciler için hazırlanmış, ev aletlerinin enerji tüketimini geçmiş ölçümlerden tahmin etmeyi öğreten uygulamalı bir projedir.

> **Proje kapsamı:** Bu bir eğitim ve geçmiş veri demosudur. Gerçek zamanlı sensör, sayaç veya ev otomasyon sistemi bağlantısı yoktur.

## Amaç

Geçmiş ölçümlerden hareketle, tahmin anı **t** iken bir sonraki 10 dakikalık aralığın (**t + 10 dakika**) `Appliances` enerji tüketimini Wh cinsinden tahmin etmek. Tahmin, kullanıcının seçtiği eşiği aşarsa Streamlit ekranında uyarı gösterilir.

## Veri kaynağı

- **Veri seti:** [Appliances Energy Prediction — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)
- **Dosya:** `energydata_complete.csv`
- **Boyut:** 19.735 gözlem; 10 dakikalık aralıklarla yaklaşık 4,5 aylık kayıt.
- **Ölçüm:** `Appliances` hedefi 10 dakikalık ölçüm aralığındaki enerji kullanımını Wh cinsinden verir.
- **Toplama ortamı:** UCI açıklamasına göre ölçümler tek bir düşük enerjili bina/ev ortamından alınmıştır.

### Tek evden gelen verinin sınırlaması

Bu veri seti yalnızca tek bir ev/bina ortamını temsil eder. Bu nedenle:

- Modelin öğrendiği tüketim alışkanlıkları, cihazlar, bina özellikleri ve yerel koşullar başka evlere genellenemez.
- Bu projedeki kronolojik train/validation/test ayrımı aynı veri kaynağındaki farklı zaman dilimlerini ölçer; farklı evlerde başarıyı kanıtlamaz.
- Başka evlerde kullanılmadan önce o evlerden veri toplanmalı, yeniden doğrulama yapılmalı ve modelin performansı ayrıca ölçülmelidir.
- Yaklaşık 4,5 aylık kayıt, mevsimler ve uzun dönemli değişimler hakkında sınırlı bilgi verir.

Sonuçları gerçek bir ev için garanti, fatura tahmini veya güvenlik alarmı olarak yorumlamayın. Streamlit ekranı yalnızca ayrılmış geçmiş test kayıtlarını gösteren bir demodur.

## Öğrenci modülleri

| Modül | İçerik | Durum |
|---|---|---|
| `01_Fundamentals` | Python temelleri, pandas ile CSV okuma ve basit grafikler — `notebooks/01_Fundamentals.ipynb` | Temel |
| `02_OOP` | Sınıf, nesne, özellik ve metot — `notebooks/02_OOP.ipynb` | Temel |
| `03_Data_Analysis` | Saatlik/günlük tüketim, grafikler ve keşifsel analiz — `notebooks/03_Data_Analysis.ipynb` | Temel |
| `04_Machine_Learning` | Özellik üretimi, kronolojik veri ayrımı, Baseline, Linear Regression, Random Forest, MAE/RMSE — `docs/04_machine_learning.md` | Temel |
| `05_Deep_Learning` | İsteğe bağlı ileri çalışma | **Opsiyonel** |
| `06_AI_Application` | Hazırlanmış modeli yükleyen tek ekranlı Streamlit tahmin demosu | Temel uygulama |
| Proje dokümantasyonu | Kurulum, veri, modelleme ve kullanım yönergeleri — `README.md`, `docs/` | Referans |

**05_Deep_Learning zorunlu değildir.** Bu proje akışını tamamlamak, mevcut makine öğrenmesi modellerini eğitmek veya 06 uygulamasını çalıştırmak için derin öğrenme modülüne ihtiyaç yoktur. İlk sürümde yeni bir derin öğrenme modeli tanımlanmamıştır.

## Proje yapısı

```text
home-energy-forecast-lab/
├── 06_AI_Application/
│   ├── README.md
│   ├── prepare_demo.py
│   └── streamlit_app.py
├── data/
│   └── raw/
│       └── energydata_complete.csv
├── docs/
│   └── 04_machine_learning.md
├── notebooks/
│   ├── 01_Fundamentals.ipynb
│   ├── 02_OOP.ipynb
│   └── 03_Data_Analysis.ipynb
├── src/
│   └── energy_forecast/
│       ├── data_analysis/
│       ├── machine_learning/
│       ├── oop/
│       └── prediction/
├── tests/
├── requirements.txt
└── README.md
```

Ham veri ve üretilen model dosyaları `.gitignore` ile Git dışında tutulur.

## Kurulum

Python 3.10 veya üzeri önerilir. Komutları depo ana klasöründe çalıştırın.

### Windows PowerShell

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PowerShell sanal ortamı etkinleştirmeyi engellerse, yalnızca mevcut terminal oturumu için:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Kurulumu kontrol edin:

```bash
python --version
python -m pip list
```

## Veri dosyasını hazırlama

UCI sayfasından `energydata_complete.csv` dosyasını indirin ve aşağıdaki konuma yerleştirin:

```text
data/raw/energydata_complete.csv
```

`data/raw/` klasörü yoksa oluşturun. Dosya adının ve uzantısının tam olarak bu şekilde olduğundan emin olun. Ham veri dosyasını Git'e eklemeyin.

## Testleri çalıştırma

Testler veri yükleyiciyi, t → t+10 dakika hedef eşleştirmesini, kronolojik bölmeyi, baseline/metrikleri ve eşik uyarısını kontrol eder.

### Windows PowerShell

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m pytest -q
```

### macOS / Linux

```bash
PYTHONPATH=src python -m pytest -q
```

Başarılı çalıştırmada pytest test sayısını ve `passed` sonucunu gösterir.

## Makine öğrenmesi modellerini karşılaştırma

Bu komut Baseline, Linear Regression ve Random Forest modellerini karşılaştırır. CSV dosyası önceden `data/raw/energydata_complete.csv` konumunda olmalıdır.

### Windows PowerShell

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m energy_forecast.machine_learning.model_comparison
```

### macOS / Linux

```bash
PYTHONPATH=src python -m energy_forecast.machine_learning.model_comparison
```

Veri kronolojik olarak eğitim, doğrulama ve test bölümlerine ayrılır. Öğrenen modeller yalnızca eğitim verisinde eğitilir; model seçimi doğrulama MAE değerine göre yapılır. Test verisi son performans raporu içindir, model seçmek için kullanılmaz. Ayrıntılar: [04 — Machine Learning](docs/04_machine_learning.md).

## 06_AI_Application: modeli hazırla ve demo ekranını aç

06 uygulaması, eğitim ve model seçimini Streamlit açılışında yapmaz. Önce aşağıdaki hazırlama komutunu çalıştırın. Bu komut üç mevcut yöntemi eğitir, doğrulama MAE ile seçim yapar ve seçilen modeli/test kayıtlarını yerel `models/energy_forecast_demo.joblib` dosyasına kaydeder.

```bash
python 06_AI_Application/prepare_demo.py
```

Ardından Streamlit uygulamasını başlatın:

```bash
python -m streamlit run 06_AI_Application/streamlit_app.py
```

Uygulamada:
1. Ayrılmış test bölümünden bir kayıt seçin.
2. Tahmin anını, hedef zamanını ve Wh cinsinden tahmini inceleyin.
3. Uyarı eşiğini belirleyin. Tahmin eşikten **büyükse** “Yüksek tüketim bekleniyor” uyarısı görünür.
4. Gerçekleşen tüketim ve mutlak hata yalnızca **Gerçekleşen değeri göster** düğmesiyle açılır.

Bu uygulama geçmiş kayıtlar üzerinde çalışan bir tahmin demosudur; canlı izleme yapmaz. Model dosyası eksikse ekran veri dosyasını yerleştirme ve `prepare_demo.py` komutunu çalıştırma yönergelerini gösterir.

06 modülünün ayrıntılı yönergeleri: [06_AI_Application/README.md](06_AI_Application/README.md).

## Modelleme ve değerlendirme notları

- Tahmin hedefi: `target_appliances_wh`, yani t+10 dakikadaki `Appliances` ölçümü.
- Özellikler tahmin anında veya öncesinde erişilebilir ölçümlerden oluşturulur.
- `target_time` ve hedef sütunu model girdisi değildir.
- Veri bölme kronolojik yapılır; karıştırma uygulanmaz.
- Metrikler Wh cinsinden MAE ve RMSE'dir.
- Bu aşamada yeni bir model eklenmemiştir. Karşılaştırılan yöntemler Baseline, Linear Regression ve Random Forest'tır.

## Katkı ve çalışma ilkeleri

- Tekrar kullanılabilir Python kodu `src/` altında tutulur.
- Testler `tests/` altında tutulur.
- Öğrenci açıklamaları Türkçe; dosya, klasör ve fonksiyon adları İngilizcedir.
- Ham veri, sanal ortam, önbellek ve üretilen model dosyaları Git'e eklenmez.
