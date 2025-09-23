from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from models import ModelManager
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

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
    family_history: bool
    obesity: bool
    alcohol_consumption: bool
    previous_heart_problems: bool
    medication_use: bool
    diet: int
    stress_level

