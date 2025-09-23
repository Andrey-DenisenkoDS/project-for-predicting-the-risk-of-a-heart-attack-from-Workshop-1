# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from models import ModelManager
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

app = FastAPI()

# Инициализация менеджера моделей
model_manager = ModelManager()

# Загрузка моделей при старте приложения
@app.on_event("startup")
async def startup_event():
    try:
        app.state.models = model_manager.load_models()
        app.state.scaler = model_manager.load_scaler()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки моделей: {str(e)}")

# Определение схемы входных данных
class InputData(BaseModel):
    diabetes: bool
    income: float
    gender: str
    family_history: bool
    obesity: bool
    alcohol_consumption: bool
    previous_heart_problems: bool
    medication_use: bool
    diet: int
    stress_level: int
    physical_activity_days_per_week: int
    age: float
    cholesterol: float
    heart_rate: float
    exercise_hours_per_week: float
    sedentary_hours_per_day: float
    bmi: float
    triglycerides: float
    sleep_hours_per_day: float
    blood_sugar: float
    ck_mb: float
    troponin: float
    systolic_blood_pressure: float
    diastolic_blood_pressure: float

class PredictionResponse(BaseModel):
    random_forest: float
    gradient_boosting: float
    xgb: float
    catboost: float

# Функция препроцессинга данных
def preprocess_data(data: InputData, scaler: StandardScaler) -> np.ndarray:
    # Преобразуем данные в DataFrame
    df = pd.DataFrame([data.dict()])
    
    # Применяем скалер
    processed_data = scaler.transform(df)
    
    return processed_data

# Эндпоинт для предсказаний
@app.post("/predict/", response_model=PredictionResponse)
async def predict_endpoint(data: InputData):
    try:
        # Получаем модели и скалер
        models = app.state.models
        scaler = app.state.scaler
        
        # Препроцессинг данных
        processed_data = preprocess_data(data, scaler)
        
        # Получение предсказаний
        rf_pred = models['random_forest'].predict_proba(processed_data)[:, 1]
        gb_pred = models['gradient_boosting'].predict_proba(processed_data)[:, 1]
        xgb_pred = models['xgb'].predict_proba(processed_data)[:, 1]
        catboost_pred = models['catboost'].predict_proba(processed_data)[:, 1]
        
        return {
            "random_forest": float(rf_pred[0]),
            "gradient_boosting": float(gb_pred[0]),
            "xgb": float(xgb_pred[0]),
            "catboost": float(catboost_pred[0])
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при предсказании: {str(e)}")

# Документация
@app.get("/")
async def read_root():
    return {"message": "Добро пожаловать в API прогнозирования"}

# Запуск приложения
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

