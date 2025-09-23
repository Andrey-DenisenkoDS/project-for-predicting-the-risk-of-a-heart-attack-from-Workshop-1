from fastapi import FastAPI, HTTPException
from models import ModelManager
from typing import Dict

app = FastAPI()

# Создаем менеджер моделей
model_manager = ModelManager()

# Функция запуска приложения
async def startup_event():
    try:
        # Пытаемся загрузить модели
        models = model_manager.load_models()
        if models is None:
            raise HTTPException(
                status_code=500,
                detail="Не удалось загрузить модели машинного обучения"
            )
        
        # Сохраняем модели в состояние приложения
        app.state.models = models
        
        # Загружаем скалер
        scaler = model_manager.load_scaler()
        if scaler is None:
            raise HTTPException(
                status_code=500,
                detail="Не удалось загрузить скалер"
            )
        
        app.state.scaler = scaler
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при инициализации: {str(e)}"
        )

# Регистрируем функцию запуска
app.add_event_handler("startup", startup_event)

# Пример эндпоинта для проверки работы
@app.get("/health")
async def health_check():
    return {"status": "OK", "message": "Сервер работает"}

# Пример эндпоинта для предсказания
@app.post("/predict")
async def predict(data: Dict):
    try:
        # Получаем модели из состояния
        models = app.state.models
        scaler = app.state.scaler
        
        # Здесь должен быть код обработки данных
        # Для примера просто возвращаем входные данные
        return {"result": "Предсказание успешно"}
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при предсказании: {str(e)}"
        )


