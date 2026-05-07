import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class ModelTrainer:
    """
    Makine öğrenmesi modeli eğitimi ve değerlendirmesi yapan sınıf.
    Öğrenci sonucunu (Passed/Failed) tahmin eder.
    """
    def __init__(self, dataframe):
        self.data = dataframe.copy()
        self.model = LogisticRegression(random_state=42)
        self._prepare_data()

    def _prepare_data(self):
        """
        Veriyi model için hazırlar: hedef değişkeni sayısala çevirir.
        """
        # result sütununu sayısala çevir: Passed=1, Failed=0
        self.data['result_numeric'] = self.data['result'].map({'Passed': 1, 'Failed': 0})

        # Gereksiz sütunları çıkar
        self.X = self.data.drop(['student_id', 'result', 'result_numeric'], axis=1)
        self.y = self.data['result_numeric']

    def train_model(self):
        """
        Modeli eğitir.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        self.model.fit(X_train, y_train)
        print("\nModel başarıyla eğitildi.")
        return X_train, X_test, y_train, y_test

    def evaluate_model(self, X_test, y_test):
        """
        Modeli değerlendirir ve performans metriklerini döndürür.
        """
        predictions = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        conf_matrix = confusion_matrix(y_test, predictions)
        class_report = classification_report(y_test, predictions)

        print("\n--- Model Performans Metrikleri ---")
        print(f"Doğruluk (Accuracy): {accuracy:.2f}")
        print("\nKarmaşıklık Matrisi:")
        print(conf_matrix)
        print("\nSınıflandırma Raporu:")
        print(class_report)

        return {
            "accuracy": accuracy,
            "confusion_matrix": conf_matrix.tolist(),
            "classification_report": class_report
        }

    def predict_sample(self, sample_data):
        """
        Örnek veri için tahmin yapar.
        """
        prediction = self.model.predict([sample_data])
        result = "Passed" if prediction[0] == 1 else "Failed"
        return result