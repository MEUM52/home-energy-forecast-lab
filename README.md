# Home Energy Forecast Lab

Başlangıç düzeyindeki Yapay Zekâ Operatörlüğü öğrencileri için hazırlanmış uygulamalı bir Python projesidir. Projede, ev aletlerinin enerji tüketimi kullanılarak **gelecek 10 dakikalık enerji tüketiminin Wh cinsinden tahmin edilmesi** ve tahmin kullanıcı tarafından belirlenen eşiği aşarsa Streamlit arayüzünde uyarı gösterilmesi hedeflenir.

> **Proje durumu:** İlk sürüm geliştirme aşamasındadır. Canlı sensör veya gerçek zamanlı cihaz bağlantısı kullanılmayacaktır.

## Projenin amaçları

- UCI Appliances Energy Prediction veri setini tanımak ve analiz etmek.
- Son ölçümü gelecek için tahmin kabul eden basit bir baseline modeli oluşturmak.
- Baseline, Linear Regression ve Random Forest sonuçlarını karşılaştırmak.
- Tahminleri Wh cinsinden değerlendirmek.
- Kullanıcının belirlediği tüketim eşiği aşıldığında Streamlit ekranında uyarı göstermek.

Derin öğrenme bölümü isteğe bağlıdır; ilk sürümün çalışması için gerekli değildir.

## Kullanılacak veri seti

- **Adı:** Appliances Energy Prediction
- **Kaynak:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)
- **Kapsam:** Bir ev ortamındaki enerji tüketimi ve çevresel ölçümler.

Ham veri dosyaları `data/raw/` altında tutulacaktır. Veri seti ve türetilmiş veri dosyaları Git'e eklenmez. Veri indirme ve hazırlama adımları proje geliştikçe uygulamaya eklenecektir.

## Teknoloji ve bağımlılıklar

- Python
- NumPy ve pandas
- Matplotlib ve Seaborn
- scikit-learn
- Streamlit
- ucimlrepo
- Jupyter ve pytest

## Kurulum

Python 3.10 veya üzeri bir sürüm kullanılması önerilir.

### Windows PowerShell

Proje klasöründe aşağıdaki komutları sırayla çalıştırın:

```powershell
py -3 -m venv .venv
..venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PowerShell sanal ortamı etkinleştirmeyi engellerse, yalnızca mevcut terminal oturumu için şu komutu çalıştırıp etkinleştirme komutunu tekrar deneyebilirsiniz:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### macOS / Linux

Proje klasöründe aşağıdaki komutları sırayla çalıştırın:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Kurulumu doğrulamak için:

```bash
python --version
python -m pip list
```

## Uygulamayı çalıştırma

Streamlit uygulamasının başlangıç dosyası:

```bash
streamlit run app/streamlit_app.py
```

> Başlangıç dosyası henüz oluşturulmadı. Uygulama dosyaları sonraki geliştirme aşamalarında eklenecektir.

## Proje aşamaları

1. **Fundamentals:** Python temelleri
2. **OOP:** Nesne yönelimli programlama
3. **Data Analysis:** Veri yükleme, temizleme ve keşifsel analiz
4. **Machine Learning:** Baseline, Linear Regression ve Random Forest
5. **Deep Learning:** İsteğe bağlı derin öğrenme denemeleri
6. **AI Application:** Tahmin bileşeni, eşik kontrolü ve Streamlit arayüzü
7. **Project Doc:** Türkçe proje ve öğrenci dokümantasyonu

## Kod ve dokümantasyon ilkeleri

- Tekrar kullanılabilir Python kodu `src/` altında tutulur.
- Testler `tests/` altında tutulur.
- Öğrenciye yönelik açıklamalar Türkçe yazılır.
- Dosya, klasör ve fonksiyon adları İngilizce kullanılır.
- İlk sürümde canlı sensör entegrasyonu yapılmaz.

## Git'e eklenmemesi gereken dosyalar

`.gitignore` aşağıdakileri Git dışında tutar:

- `.venv/` ve diğer sanal ortam klasörleri
- Ham ve işlenmiş veri dosyaları (`data/raw/`, `data/processed/`)
- Jupyter notebook önbellekleri
- Üretilen model dosyaları (`models/` ve yaygın model dosyası uzantıları)
- Python ve test önbellekleri ile yerel ortam dosyaları

Klasörlerin Git'te görünür kalması için ilgili `.gitkeep` dosyaları kullanılabilir.
