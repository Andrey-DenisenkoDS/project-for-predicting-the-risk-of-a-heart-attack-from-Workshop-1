from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
from joblib import dump, load
from dotenv import load_dotenv
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Загружаем переменные окружения
load_dotenv()

# Получаем базовый путь к проекту
BASE_DIR = Path(__file__).parent.resolve()

# Формируем путь к папке с моделями
PRO_PATH = os.getenv('PRO_PATH', 'app/models')
MODEL_DIR = BASE_DIR / PRO_PATH
os.makedirs(MODEL_DIR, exist_ok=True)

# Определяем пути к конкретным моделям
RF_MODEL_PATH = MODEL_DIR / 'rf_model.pkl'
GB_MODEL_PATH = MODEL_DIR / 'gb_model.pkl'
XGB_MODEL_PATH = MODEL_DIR / 'xgb_model.pkl'
CATBOOST_MODEL_PATH = MODEL_DIR / 'catboost_model.pkl'
SCALER_PATH = MODEL_DIR / 'scaler.pkl'

class ModelManager:
    def __init__(self):
        self.rf_model = None
        self.gb_model = None
        self.xgb_model = None
        self.catboost_model = None
        self.scaler = None

    def create_models(self, n_estimators=100, random_state=42):
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        )
        
        gb = GradientBoostingClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        )
        
        xgb = xgb.XGBClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        )
        
        catboost = CatBoostClassifier(
            iterations=n_estimators,
            random_seed=random_state,
            verbose=False
        )
        
        return rf, gb, xgb, catboost

    def save_models(self, rf, gb, xgb_model, catboost):
        dump(rf, RF_MODEL_PATH)
        dump(gb, GB_MODEL_PATH)
        dump(xgb_model, XGB_MODEL_PATH)
        catboost.save_model(CATBOOST_MODEL_PATH)

    def load_models(self):
        try:
            rf = load(RF_MODEL_PATH)
            gb = load(GB_MODEL_PATH)
            xgb = load(XGB_MODEL_PATH)
            catboost = CatBoostClassifier()
            catboost.load_model(CATBOOST_MODEL_PATH)
            return {
                'random_forest': rf,
                'gradient_boosting': gb,
                'xgb': xgb,
                'catboost': catboost
            }
        except FileNotFoundError as e:
            print(f"Ошибка загрузки модели: {e}")
            return None

    def load_scaler(self):
        try:
            return load(SCALER_PATH)
        except FileNotFoundError:
            print(f"Ошибка: файл скалера не найден по пути {SCALER_PATH}")
            return None

    def save_scaler(self, scaler):
        dump(scaler, SCALER_PATH)

    def train_and_save(self, X_train, y_train):
        # Создаем модели
        rf, gb, xgb_model, catboost = self.create_models()
        
        # Обучаем скалер
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Обучаем модели
        rf.fit(X_train_scaled, y_train)
        gb.fit(X_train_scaled, y_train)
        xgb_model.fit(X_train_scaled, y_train)
        catboost.fit(X_train_scaled, y_train)
        
        # Сохраняем все
        self.save


