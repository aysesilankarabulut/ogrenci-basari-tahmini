import matplotlib.pyplot as plt
import os
from pathlib import Path

class Visualizer:
    """
    Veri görselleştirme işlemlerini gerçekleştiren sınıf.
    Matplotlib kullanarak grafikler oluşturur ve kaydeder.
    """
    def __init__(self, dataframe, output_dir):
        self.data = dataframe
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_final_score_histogram(self):
        """
        Final notlarının histogramını oluşturur.
        """
        plt.figure(figsize=(8, 5))
        plt.hist(self.data["final_score"], bins=8, edgecolor='black')
        plt.title("Final Notlarının Dağılımı")
        plt.xlabel("Final Notu")
        plt.ylabel("Öğrenci Sayısı")
        plt.savefig(self.output_dir / "final_notlari_histogram.png")
        plt.close()
        return "Final notlarının dağılımı histogramı oluşturuldu."

    def create_study_hours_scatter(self):
        """
        Çalışma saati ve final notu ilişkisini scatter plot ile gösterir.
        """
        plt.figure(figsize=(8, 5))
        plt.scatter(self.data["study_hours"], self.data["final_score"], alpha=0.7)
        plt.title("Çalışma Saati ve Final Notu İlişkisi")
        plt.xlabel("Haftalık Çalışma Saati")
        plt.ylabel("Final Notu")
        plt.savefig(self.output_dir / "calisma_saati_final_notu.png")
        plt.close()
        return "Çalışma saati ve final notu ilişkisi scatter plot'u oluşturuldu."

    def create_result_bar_chart(self):
        """
        Geçti/Kaldı dağılımını bar chart ile gösterir.
        """
        plt.figure(figsize=(8, 5))
        result_counts = self.data["result"].value_counts()
        plt.bar(result_counts.index, result_counts.values, color=['green', 'red'])
        plt.title("Geçti / Kaldı Dağılımı")
        plt.xlabel("Sonuç")
        plt.ylabel("Öğrenci Sayısı")
        plt.savefig(self.output_dir / "sonuc_dagilimi_bar.png")
        plt.close()
        return "Geçti/Kaldı dağılımı bar chart'ı oluşturuldu."

    def create_all_graphs(self):
        """
        Tüm grafikleri oluşturur ve yorumlarını döndürür.
        """
        comments = []
        comments.append(self.create_final_score_histogram())
        comments.append(self.create_study_hours_scatter())
        comments.append(self.create_result_bar_chart())
        print("\nGrafikler başarıyla oluşturuldu ve outputs/graphs klasörüne kaydedildi.")
        return comments