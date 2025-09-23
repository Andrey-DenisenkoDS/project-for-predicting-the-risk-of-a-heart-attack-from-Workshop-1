from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, List, Optional
from sklearn.preprocessing import StandardScaler

# Определяем схему входных данных
class PredictionRequest(BaseModel):
    age: float
    cholesterol: float
    heart_rate: float
    diabetes: int
    family_history: int
    smoking: int
    obesity: int
    alcohol_consumption: float
    exercise_hours_per_week: float
    diet: float
    previous_heart_problems: int
    medication_use: int
    stress_level: float
    sedentary_hours_per_day: float
    income: float
    bmi: float
    triglycerides: float
    physical_activity_days_per_week: float
    sleep_hours_per_day: float
    blood_sugar: float
    ck_mb: float
    troponin: float
    systolic_blood_pressure: float
    diastolic_blood_pressure: float
    gender: str

class ModelManager:
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.threshold = 0.6

    def find_model_path(self, base_name: str) -> Optional[str]:
        search_paths = [
            os.path.join('models', base_name),
            os.path.join('app', 'models', base_name),
            os.path.join('C:\\Users\\seaho\\Workshop One\\models', base_name),
            os.path.join('C:\\Users\\seaho\\Workshop One\\app\\models', base_name)
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                return path
        return None

    def load_models(self) -> Dict:
        model_names = {
            'catboost': 'catboost_best_model.pkl',
            'xgb': 'xgb_model.pkl',
            'lightgbm': 'lightgbm_model.pkl',
            'rf': 'rf_model.pkl'
        }
        
        for model_name, base_name in model_names.items():
            path = self.find_model_path(base_name)
            if path:
                try:
                    self.models[model_name] = joblib.load(path)
                    print(f"Модель {model_name} успешно загружена из {path}")
                except Exception as e:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Ошибка загрузки модели {model_name}: {str(e)}"
                    )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Модель {model_name} не найдена в доступных директориях"
                )
        return self.models

    def load_scaler(self) -> Optional[StandardScaler]:
        scaler_paths = [
            os.path.join('models', 'scaler.pkl'),
            os.path.join('app', 'models', 'scaler.pkl'),
            os.path.join('C:\\Users\\seaho\\Workshop One\\models', 'scaler.pkl'),
            os.path.join('C:\\Users\\seaho\\Workshop One\\app\\models', 'scaler.pkl')
        ]
        
        for path in scaler_paths:
            if os.path.exists(path):
                try:
                    self.scaler = joblib.load(path)
                    print(f"Скалер успешно загружен из {path}")
                    return self.scaler
                except Exception as e:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Ошибка загрузки скалера: {str(e)}"
                    )
        return None

    
    def preprocess_data(self, data: Dict) -> pd.DataFrame:
        try:
            df = pd.DataFrame([data])
            
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
            
            # Применение скалера к числовым признакам
            df[numeric_features] = self.scaler.transform(df[numeric_features])
            
            # One-hot encoding для категориальных признаков
            df = pd.get_dummies(df, columns=categorical_features, drop_first=True)
            
            return df
        
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Ошибка при предобработке данных: {str(e)}"
            )

    def predict(self, data: Dict) -> Dict:
        try:
            # Предобработка данных
            processed_data = self.preprocess_data(data)
            
            # Получение предсказаний от каждой модели
            predictions = {}
            for model_name, model in self.models.items():
                try:
                    preds = model.predict_proba(processed_data)[:, 1]
                    predictions[model_name] = preds[0]
                except Exception as e:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Ошибка при предсказании моделью {model_name}: {str(e)}"
                    )
            
            # Агрегация предсказаний (среднее значение)
            ensemble_proba = np.mean(list(predictions.values()))
            ensemble_pred = int(ensemble_proba >= self.threshold)
            
            return {
                "probability": float(ensemble_proba),
                "prediction": ensemble_pred,
                "model_predictions": predictions
            }
        
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Ошибка при предсказании: {str(e)}"
            )

# В файле main.py добавляем эндпоинт для предсказания
@app.post("/predict")
async def predict(data: PredictionRequest):
    try:
        # Получаем модели и скалер из состояния
        models = app.state.models
        scaler = app.state.scaler
        
        # Создаем экземпляр менеджера моделей
        model_manager = ModelManager()
        model_manager.models = models
        model_manager.scaler = scaler
        
        # Выполняем предсказание
        result = model_manager.predict(data.dict())
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при предсказании: {str(e)}"
        )
            
