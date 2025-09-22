from fastapi import FastAPI, UploadFile, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import logging
import uvicorn
import argparse
from model import Model
import os

# Инициализация приложения
app = FastAPI()
app.mount("/tmp", StaticFiles(directory="tmp"), name='images')
templates = Jinja2Templates(directory="templates")

# Создаем директорию для временных файлов, если её нет
if not os.path.exists("tmp"):
    os.makedirs("tmp")

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s %(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# Health check
@app.get("/health")
def health():
    return {"status": "OK"}

# Главная страница
@app.get("/")
def main(request: Request):
    return templates.TemplateResponse(
        "start_form.html", 
        {"request": request}
    )

# Обработка запроса
@app.post("/predict-file")
async def predict_file(file: UploadFile, request: Request):
    try:
        # Проверяем, что файл действительно загружен
        if not file.filename:
            raise HTTPException(status_code=400, detail="Файл не выбран")
        
        save_pth = f"tmp/{file.filename}"
        logger.info(f'Processing file - {save_pth}')
        
        # Сохранение файла
        with open(save_pth, "wb") as fid:
            fid.write(await file.read())
            
        # Предсказание
        predictor = Model(0.6)
        status, result = predictor(save_pth)
        
        return templates.TemplateResponse(
            "res_form.html", 
            {
                "request": request,
                "res": status,
                "message": status,
                "path": result
            }
        )
        
    except FileNotFoundError:
        logger.error("Файл не найден")
        raise HTTPException(status_code=404, detail="Файл не найден")
        
    except Exception as e:
        logger.error(f"Ошибка обработки: {e}")
        raise HTTPException(status_code=500, detail="Ошибка обработки файла")

# Запуск сервера
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=8010, type=int)
    parser.add_argument("--host", default="127.0.0.8", type=str)
    args = parser.parse_args()
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=True
    )
