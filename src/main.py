from pathlib import Path
from data_loader import DataLoader
from analyzer import DataAnalyzer
from visualizer import Visualizer
from model_trainer import ModelTrainer

def main():
    """
    Ana program akışı: veri yükleme, analiz, görselleştirme, model eğitimi ve rapor kaydetme.
    """
    # Dosya yolları
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "students_performance.csv"
    graphs_path = project_root / "outputs" / "graphs"
    reports_path = project_root / "outputs" / "reports"

    # 1. Veri yükleme
    loader = DataLoader(data_path)
    data = loader.load_data()

    if data is None:
        print("Program sonlandırılıyor: Veri yüklenemedi.")
        return

    # 2. Veri analizi
    analyzer = DataAnalyzer(data)
    analyzer.show_basic_statistics()
    numpy_stats = analyzer.calculate_numpy_statistics()
    analysis_results = analyzer.analyze_results()

    # 3. Görselleştirme
    visualizer = Visualizer(data, graphs_path)
    graph_comments = visualizer.create_all_graphs()

    # 4. Model eğitimi
    model_trainer = ModelTrainer(data)
    X_train, X_test, y_train, y_test = model_trainer.train_model()
    model_metrics = model_trainer.evaluate_model(X_test, y_test)

    # 5. Rapor kaydetme
    save_report(reports_path, data, numpy_stats, analysis_results, graph_comments, model_metrics)

def save_report(report_folder, data, numpy_stats, analysis_results, graph_comments, model_metrics):
    """
    Analiz sonuçlarını rapor dosyasına kaydeder.
    """
    report_path = report_folder / "project_report.txt"
    report_folder.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as file:
        file.write("Öğrenci Başarı Analizi Raporu\n")
        file.write("=" * 40 + "\n\n")

        file.write("1. Veri Seti Tanıtımı\n")
        file.write("-" * 20 + "\n")
        file.write(f"Toplam öğrenci sayısı: {len(data)}\n")
        file.write(f"Sütunlar: {', '.join(data.columns.tolist())}\n\n")

        file.write("2. Eksik Veri Kontrolü\n")
        file.write("-" * 20 + "\n")
        missing = data.isnull().sum()
        for col, count in missing.items():
            file.write(f"{col}: {count} eksik veri\n")
        file.write("\n")

        file.write("3. NumPy Hesaplamaları\n")
        file.write("-" * 20 + "\n")
        for key, value in numpy_stats.items():
            file.write(f"{key}: {value:.2f}\n")
        file.write("\n")

        file.write("4. Pandas Analizleri\n")
        file.write("-" * 20 + "\n")
        file.write(f"Ortalama final notu: {analysis_results['ortalama_not']:.2f}\n")
        file.write("Geçti/Kaldı dağılımı:\n")
        for result, count in analysis_results['sonuc_dagilimi'].items():
            file.write(f"  {result}: {count}\n")
        file.write("\n")

        file.write("5. Grafik Yorumları\n")
        file.write("-" * 20 + "\n")
        for comment in graph_comments:
            file.write(f"- {comment}\n")
        file.write("\n")

        file.write("6. Model Açıklaması\n")
        file.write("-" * 20 + "\n")
        file.write("Kullanılan model: Logistic Regression\n")
        file.write("Hedef değişken: Öğrenci sonucu (Passed/Failed)\n")
        file.write("Giriş özellikleri: study_hours, attendance, previous_score, sleep_hours, final_score\n\n")

        file.write("7. Model Başarı Sonucu\n")
        file.write("-" * 20 + "\n")
        file.write(f"Doğruluk (Accuracy): {model_metrics['accuracy']:.2f}\n")
        file.write("Karmaşıklık Matrisi:\n")
        for row in model_metrics['confusion_matrix']:
            file.write(f"  {row}\n")
        file.write("\nSınıflandırma Raporu:\n")
        file.write(model_metrics['classification_report'])
        file.write("\n")

        file.write("8. Genel Değerlendirme\n")
        file.write("-" * 20 + "\n")
        if analysis_results['ortalama_not'] >= 60:
            file.write("Sınıfın genel başarı seviyesi iyidir.\n")
        else:
            file.write("Sınıfın genel başarı seviyesini iyileştirmek gerekebilir.\n")
        file.write("Model performansı göz önünde bulundurulduğunda, öğrenci başarısını tahmin etmek için kullanılabilir.\n")

    print(f"\nRapor başarıyla kaydedildi: {report_path}")

if __name__ == "__main__":
    main()