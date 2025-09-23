from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List
import pandas as pd
from fastapi.responses import JSONResponse
from utils import predict, load_models, load_scaler
from schemas import PredictionResponse
import io
import os
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

app = FastAPI()

# Получаем путь к моделям из .env
PRO_PATH = os.getenv('PRO_PATH', 'app/models')

# Загружаем модели при старте приложения
MODELS = load_models()
SCALER = load_scaler()

@app.post("/predict/", response_model=List[PredictionResponse])
async def predict_endpoint(file: UploadFile = File(...)):
    try:
        # Читаем файл
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Проверяем наличие обязательного столбца 'id'
        if 'id' not in df.columns:
            return JSONResponse(content={"error": "Файл должен содержать столбец 'id'"}, status_code=400)
        
        # Делаем предсказания
        result_df = predict(df, MODELS, SCALER)
        
        # Создаем CSV файл
        csv = result_df.to_csv(index=False)
        
        # Сохраняем временный файл
        temp_file_path = 'predictions.csv'
        with open(temp_file_path, 'w') as f:
            f.write(csv)
        
        return FileResponse(temp_file_path, filename='predictions.csv')
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
async def root():
    return {"message": "Welcome to Prediction Service!"}
