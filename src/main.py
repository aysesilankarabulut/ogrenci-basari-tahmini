import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        try:
            data = pd.read_csv(self.file_path)
            print("Veri seti başarıyla yüklendi.")
            return data
        except FileNotFoundError:
            print("Hata: Veri seti dosyası bulunamadı.")
            return None
        except Exception as error:
            print("Beklenmeyen bir hata oluştu:", error)
            return None


class DataAnalyzer:
    def __init__(self, data):
        self.data = data

    def show_basic_info(self):
        print("\n--- İlk 5 Satır ---")
        print(self.data.head())

        print("\n--- Veri Seti Bilgisi ---")
        print(self.data.info())

        print("\n--- Eksik Veri Kontrolü ---")
        print(self.data.isnull().sum())

        print("\n--- Genel İstatistiksel Özet ---")
        print(self.data.describe())

    def calculate_numpy_statistics(self):
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
        print("\n--- Geçti / Kaldı Dağılımı ---")
        print(self.data["result"].value_counts())

        average_score = self.data["final_score"].mean()

        print("\n--- Genel Başarı Yorumu ---")
        if average_score >= 60:
            print("Sınıfın genel başarı ortalaması yeterli seviyededir.")
        else:
            print("Sınıfın genel başarı ortalaması düşük seviyededir.")


class Visualizer:
    def __init__(self, data, graph_folder):
        self.data = data
        self.graph_folder = graph_folder
        os.makedirs(self.graph_folder, exist_ok=True)

    def create_final_score_histogram(self):
        plt.figure(figsize=(8, 5))
        plt.hist(self.data["final_score"], bins=8)
        plt.title("Final Notlarının Dağılımı")
        plt.xlabel("Final Notu")
        plt.ylabel("Öğrenci Sayısı")
        plt.savefig(os.path.join(self.graph_folder, "final_notlari_histogram.png"))
        plt.close()

    def create_study_hours_graph(self):
        plt.figure(figsize=(8, 5))
        plt.scatter(self.data["study_hours"], self.data["final_score"])
        plt.title("Çalışma Saati ve Final Notu İlişkisi")
        plt.xlabel("Haftalık Çalışma Saati")
        plt.ylabel("Final Notu")
        plt.savefig(os.path.join(self.graph_folder, "calisma_saati_final_notu.png"))
        plt.close()

    def create_attendance_graph(self):
        plt.figure(figsize=(8, 5))
        plt.scatter(self.data["attendance"], self.data["final_score"])
        plt.title("Derse Devam Oranı ve Final Notu İlişkisi")
        plt.xlabel("Derse Devam Oranı")
        plt.ylabel("Final Notu")
        plt.savefig(os.path.join(self.graph_folder, "devam_orani_final_notu.png"))
        plt.close()

    def create_all_graphs(self):
        self.create_final_score_histogram()
        self.create_study_hours_graph()
        self.create_attendance_graph()
        print("\nGrafikler başarıyla oluşturuldu ve outputs/graphs klasörüne kaydedildi.")

class ModelTrainer:
    def __init__(self, data):
        self.data = data
        self.model = LinearRegression()

    def train_model(self):
        # Modelin kullanacağı giriş verileri
        X = self.data[["study_hours", "attendance", "previous_score", "sleep_hours"]]

        # Modelin tahmin etmeye çalışacağı hedef değer
        y = self.data["final_score"]

        # Veriyi eğitim ve test olarak ayırma
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Modeli eğitme
        self.model.fit(X_train, y_train)

        # Test verileri üzerinde tahmin yapma
        predictions = self.model.predict(X_test)

        # Performans metrikleri
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)

        print("\n--- Makine Öğrenmesi Model Sonuçları ---")
        print(f"MAE: {mae:.2f}")
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R2 Skoru: {r2:.2f}")

        print("\n--- Gerçek ve Tahmin Edilen Notlar ---")
        for real, predicted in zip(y_test, predictions):
            print(f"Gerçek Not: {real} - Tahmin Edilen Not: {predicted:.2f}")

        return {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2
        }

class ReportSaver:
    def __init__(self, data, report_folder):
        self.data = data
        self.report_folder = report_folder
        os.makedirs(self.report_folder, exist_ok=True)

    def save_text_report(self):
        report_path = os.path.join(self.report_folder, "summary_report.txt")

        with open(report_path, "w", encoding="utf-8") as file:
            file.write("Öğrenci Başarı Analizi Raporu\n")
            file.write("--------------------------------\n\n")
            file.write(f"Toplam öğrenci sayısı: {len(self.data)}\n")
            file.write(f"Ortalama final notu: {self.data['final_score'].mean():.2f}\n")
            file.write(f"En yüksek final notu: {self.data['final_score'].max()}\n")
            file.write(f"En düşük final notu: {self.data['final_score'].min()}\n\n")
            file.write("Geçti / Kaldı Dağılımı:\n")
            file.write(str(self.data["result"].value_counts()))

        print("\nRapor başarıyla kaydedildi:", report_path)


def main():
    file_path = "../data/students.csv"
    graph_folder = "../outputs/graphs"
    report_folder = "../outputs/reports"

    loader = DataLoader(file_path)
    data = loader.load_data()

    if data is not None:
       analyzer = DataAnalyzer(data)
       analyzer.show_basic_info()
       analyzer.calculate_numpy_statistics()
       analyzer.analyze_results()

       visualizer = Visualizer(data, graph_folder)
       visualizer.create_all_graphs()

       model_trainer = ModelTrainer(data)
       model_trainer.train_model()

       report_saver = ReportSaver(data, report_folder)
       report_saver.save_text_report()


if __name__ == "__main__":
    main()