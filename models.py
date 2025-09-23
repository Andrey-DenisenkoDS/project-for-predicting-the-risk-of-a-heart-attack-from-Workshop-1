from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
from joblib import dump, load
import os
from dotenv import load_dotenv
from typing import Tuple, Dict

# Загружаем переменные окружения
load_dotenv()

# Определяем пути к моделям
PRO_PATH = os.getenv('PRO_PATH', 'app/models')
RF_MODEL_PATH = os.path.join(PRO_PATH, 'rf_model.pkl')
GB_MODEL_PATH = os.path.join(PRO_PATH, 'gb_model.pkl')
XGB_MODEL_PATH = os.path.join(PRO_PATH, 'xgb_model.pkl')
CATBOOST_MODEL_PATH = os.path.join(PRO_PATH, 'catboost_model.pkl')
SCALER_PATH = os.path.join(PRO_PATH, 'scaler.pkl')

class ModelManager:
    def __init__(self):
        self.rf_model = None
        self.gb_model = None
        self.xgb_model = None
        self.catboost_model = None
        self.scaler = None

    def create_models(self, 
                     n_estimators: int = 100, 
                     random_state: int = 42) -> Tuple:
        """
        Создание моделей с заданными параметрами
        """
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

    def save_models(self, 
                   rf: RandomForestClassifier, 
                   gb: GradientBoostingClassifier, 
                   xgb_model: xgb.XGBClassifier, 
                   catboost: CatBoostClassifier):
        """
        Сохранение моделей
        """
        dump(rf, RF_MODEL_PATH)
        dump(gb, GB_MODEL_PATH)
        dump(xgb_model, XGB_MODEL_PATH)
        catboost.save_model(CATBOOST_MODEL_PATH)

    def load_models(self) -> Dict[str, object]:
        """
        Загрузка моделей
        """
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

    def load_scaler(self) -> StandardScaler:
        """
        Загрузка скалера
        """
        from sklearn.preprocessing import StandardScaler
        return load(SCALER_PATH)

# Пример использования
if __name__ == "__main__":
    manager = ModelManager()
    
    # Создание новых моделей (пример)
    rf, gb, xgb_model, catboost = manager.create_models()
    
    # Сохранение моделей (пример)
    # manager.save_models(rf, gb, xgb_model, catboost)
    
    # Загрузка моделей
    models = manager.load_models()
    scaler = manager.load_scaler()
