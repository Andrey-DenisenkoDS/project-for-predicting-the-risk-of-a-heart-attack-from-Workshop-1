# schemas.py
from pydantic import BaseModel
from typing import List

class PredictionRequest(BaseModel):
    """
    Модель для валидации входных данных
    """
    data: List[dict]  # Входящие данные для предсказания

class PredictionResult(BaseModel):
    """
    Модель для одного результата предсказания
    """
    id: int  # Идентификатор записи
    prediction: float  # Результат предсказания (вероятность)
    
    class Config:
        orm_mode = True  # Для работы с ORM моделями

class PredictionResponse(BaseModel):
    """
    Модель для ответа API
    """
    predictions: List[PredictionResult]  # Список результатов предсказаний
    status: str = "success"  # Статус выполнения
    message: str = "Predictions completed successfully"  # Сообщение

class HealthCheckResponse(BaseModel):
    """
    Модель для ответа проверки работоспособности сервиса
    """
    status: str = "ok"
    message: str = "Service is running"
    version: str = "1.0.0"
