# Öğrenci Başarı Analizi Projesi

Bu proje, öğrenci başarı verilerini analiz etmek ve tahmin modelleri geliştirmek için hazırlanmıştır.

## Gereksinimler

Programın çalışması için aşağıdaki Python kütüphanelerinin yüklü olması gerekir:

- pandas
- numpy
- matplotlib
- scikit-learn

## Kurulum

Gerekli kütüphaneleri yüklemek için:

```bash
pip install pandas numpy matplotlib scikit-learn
```

## Çalıştırma

Projeyi çalıştırmak için terminalde aşağıdaki komutu kullanın:

```bash
python src/main.py
```

## Çıktılar

Program çalıştırıldığında aşağıdaki çıktılar üretilir:

- **Grafikler**: `outputs/graphs/` klasöründe
  - final_notlari_histogram.png
  - calisma_saati_final_notu.png
  - sonuc_dagilimi_bar.png

- **Rapor**: `outputs/reports/project_report.txt`
  - Veri analizi sonuçları
  - Model performans metrikleri
  - Genel değerlendirme

## Proje Yapısı

```
project/
│
├── data/
│   └── students_performance.csv
│
├── src/
│   ├── main.py
│   ├── data_loader.py
│   ├── analyzer.py
│   ├── visualizer.py
│   └── model_trainer.py
│
├── outputs/
│   ├── graphs/
│   └── reports/
│
└── README.md
```