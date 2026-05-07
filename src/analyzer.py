import pandas as pd
import numpy as np

class DataAnalyzer:
    """
    Veri analizi işlemlerini gerçekleştiren sınıf.
    Pandas ve NumPy kullanarak istatistiksel analizler yapar.
    """
    def __init__(self, dataframe):
        self.data = dataframe

    def show_basic_statistics(self):
        """
        Veri setinin temel istatistiklerini gösterir.
        """
        print("\n--- Genel İstatistiksel Özet ---")
        print(self.data.describe())

        print("\n--- Eksik Veri Kontrolü ---")
        print(self.data.isnull().sum())

    def calculate_numpy_statistics(self):
        """
        NumPy kullanarak final_score sütunu için istatistikler hesaplar.
        """
        final_scores = self.data["final_score"].to_numpy()

        statistics = {
            "Ortalama Final Notu": np.mean(final_scores),
            "En Yüksek Final Notu": np.max(final_scores),
            "En Düşük Final Notu": np.min(final_scores),
            "Standart Sapma": np.std(final_scores)
        }

        print("\n--- NumPy ile Hesaplanan İstatistikler ---")
        for title, value in statistics.items():
            print(f"{title}: {value:.2f}")

        return statistics

    def analyze_results(self):
        """
        Geçti/Kaldı dağılımını analiz eder ve korelasyon hesaplar.
        """
        print("\n--- Geçti / Kaldı Dağılımı ---")
        print(self.data["result"].value_counts())

        # Korelasyon matrisi
        numeric_data = self.data.select_dtypes(include=[np.number])
        correlation = numeric_data.corr()
        print("\n--- Korelasyon Matrisi ---")
        print(correlation)

        # Genel başarı yorumu
        average_score = self.data["final_score"].mean()
        print("\n--- Genel Başarı Yorumu ---")
        if average_score >= 60:
            print("Sınıfın genel başarı ortalaması yeterli seviyededir.")
        else:
            print("Sınıfın genel başarı ortalaması düşük seviyededir.")

        return {
            "ortalama_not": average_score,
            "korelasyon": correlation.to_dict(),
            "sonuc_dagilimi": self.data["result"].value_counts().to_dict()
        }