from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List
import pandas as pd
import io
import os
import tempfile
from dotenv import load_dotenv
from utils import predict, load_models, load_scaler
from schemas import PredictionResponse
import logging
from models import  # Убедитесь, что путь правильный
from schemas import  # Убедитесь, что путь правильный
from utils import predict, load_models, load_scaler  # Исправленный импорт

# Настройка логирования
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Загружаем переменные окружения
load_dotenv()

app = FastAPI(
    title="Prediction Service",
    description="API для предсказаний с использованием различных моделей машинного обучения",
    version="1.0.0"
)

# Получаем путь к моделям из .env
PRO_PATH = os.getenv('PRO_PATH', 'app/models')

# Загружаем модели при старте приложения
MODELS = load_models()
SCALER = load_scaler()

def remove_file(file_path: str):
    """Функция для удаления временного файла"""
    try:
        os.remove(file_path)
    except OSError as e:
        logger.error(f"Ошибка при удалении файла {file_path}: {str(e)}")

@app.post(
    "/predict/",
    response_model=List[PredictionResponse],
    summary="Сделать предсказания по загруженным данным"
)
async def predict_endpoint(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks
):
    """
    Эндпоинт для загрузки CSV файла и получения предсказаний
    
    Параметры:
    - file: CSV файл с данными для предсказания
    
    Возвращает:
    - CSV файл с результатами предсказания
    """
    try:
        logger.info("Получен запрос на предсказание")
        
        # Читаем файл
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Проверяем наличие обязательного столбца 'id'
        if 'id' not in df.columns:
            return JSONResponse(
                content={"error": "Файл должен содержать столбец 'id'"},
                status_code=400
            )
        
        # Делаем предсказания
        result_df = predict(df, MODELS, SCALER)
        
        # Создаем временный файл
        with tempfile.NamedTemporaryFile(
            delete=False, 
            suffix='.csv', 
            mode='w'
        ) as temp_file:
            result_df.to_csv(temp_file, index=False)
            temp_file_path = temp_file.name
        
        # Добавляем задачу на удаление файла в фоновом режиме
        background_tasks.add_task(remove_file, temp_file_path)
        
        return FileResponse(
            temp_file_path, 
            filename='predictions.csv',
            background=background_tasks
        )
    
    except Exception as e:
        logger.error(f"Произошла ошибка: {str(e)}")
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

@app.get(
    "/",
    summary="Проверка работоспособности сервиса"
)
async def root():
    """
    Корневой эндпоинт для проверки работы сервиса
    """
    return {"message": "Welcome to Prediction Service!"}

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Глобальный обработчик исключений
    """
    logger.error(f"Необработанное исключение: {str(exc)}")
    return JSONResponse(
        content={"error": "Произошла внутренняя ошибка сервера"},
        status_code=500
    )

