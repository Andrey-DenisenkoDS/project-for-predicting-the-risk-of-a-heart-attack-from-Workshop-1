import pandas as pd
import numpy as np
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from catboost import CatBoostClassifier

class Model:
    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold
        self.pipeline = None
        self.load_model()
        
    def load_model(self):
        try:
            # Загружаем сохраненный пайплайн с моделью
            self.pipeline = joblib.load('models/catboost_best_model.pkl')
        except FileNotFoundError:
            raise Exception("Модель не найдена. Проверьте путь к файлу модели.")
            
    def preprocess_data(self, file_path: str) -> pd.DataFrame:
        try:
            # Читаем данные из файла с вашими настройками
            df = pd.read_csv(
                file_path,
                sep=',', 
                encoding='utf-8', 
                decimal=','
            )
            
            # Определяем признаки
            numeric_features = [
                'age', 
                'cholesterol', 
                'heart_rate', 
                'diabetes', 
                'family_history', 
                'smoking', 
                'obesity', 
                'alcohol_consumption', 
                'exercise_hours_per_week', 
                'diet', 
                'previous_heart_problems', 
                'medication_use', 
                'stress_level', 
                'sedentary_hours_per_day', 
                'income', 
                'bmi', 
                'triglycerides', 
                'physical_activity_days_per_week', 
                'sleep_hours_per_day', 
                'blood_sugar', 
                'ck_mb', 
                'troponin', 
                'systolic_blood_pressure', 
                'diastolic_blood_pressure'
            ]
            
            categorical_features = ['gender']
            
            # Обработка пропущенных значений
            df[numeric_features] = df[numeric_features].fillna(df[numeric_features].mean())
            
            # Проверка наличия всех необходимых признаков
            missing_features = set(numeric_features + categorical_features) - set(df.columns)
            if missing_features:
                raise ValueError(f"Отсутствуют обязательные признаки: {missing_features}")
                
            return df
            
        except FileNotFoundError:
            raise Exception(f"Файл не найден: {file_path}")
        except Exception as e:
            raise Exception(f"Ошибка при чтении файла: {str(e)}")
            
    def predict(self, file_path: str):
        try:
            # Предобработка данных
            data = self.preprocess_data(file_path)
            
            # Получение предсказаний
            predictions = self.pipeline.predict_proba(data)[:, 1]
            
            # Бинаризация предсказаний по заданному порогу
            binary_predictions = (predictions >= self.threshold).astype(int)
            
            # Сохраняем результаты
            result_path = f"results/{os.path.basename(file_path)}_predictions.csv"
            os.makedirs(os.path.dirname(result_path), exist_ok=True)
            pd.DataFrame({
                'probability': predictions,
                'prediction': binary_predictions
            }).to_csv(result_path, index=False)
            
            return "success", result_path
            
        except Exception as e:
            raise Exception(f"Ошибка при предсказании: {str(e)}")
            
    def __call__(self, file_path: str):
        return self.predict(file_path)

if __name__ == "__main__":
    try:
        model = Model(0.6)
        status, result_path = model("data/example.csv")
        print(f"Статус: {status}, Путь к результатам: {result_path}")
    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")

