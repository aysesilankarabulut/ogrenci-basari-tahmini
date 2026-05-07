import pandas as pd
import numpy as np
from pathlib import Path

class DataLoader:
    """
    Veri yükleme ve temel temizleme işlemlerini gerçekleştiren sınıf.
    CSV dosyasını okur, eksik verileri kontrol eder ve temizler.
    """
    def __init__(self, file_path):
        self.file_path = Path(file_path)

    def load_data(self):
        """
        CSV dosyasını yükler ve eksik verileri kontrol eder.
        """
        try:
            data = pd.read_csv(self.file_path)
            print("Veri seti başarıyla yüklendi.")
            return self._clean_data(data)
        except FileNotFoundError:
            print("Hata: Veri seti dosyası bulunamadı.")
            return None
        except Exception as error:
            print("Beklenmeyen bir hata oluştu:", error)
            return None

    def _clean_data(self, data):
        """
        Eksik verileri kontrol eder ve gerekirse doldurur veya kaldırır.
        """
        print("\n--- Eksik Veri Kontrolü ---")
        missing_data = data.isnull().sum()
        print(missing_data)

        # Eksik verileri ortalama ile doldur (sayısal sütunlar için)
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

        # Temel bilgiler
        print("\n--- Veri Seti Bilgisi ---")
        print(data.info())

        print("\n--- İlk 5 Satır ---")
        print(data.head())

        return data