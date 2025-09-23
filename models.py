from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
from joblib import dump, load
import os
from dotenv import load_dotenv
from typing import Tuple, Dict
from sklearn.preprocessing import StandardScaler

# Загружаем переменные окружения
load_dotenv()

# Создаем путь к папке models, если её нет
MODEL_DIR = 'models'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Определяем пути к моделям
PRO_PATH = os.getenv('PRO_PATH', MODEL_DIR)  # Используем созданную папку
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
        # Создаем Random Forest классификатор
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1
        )
        
        # Создаем Gradient Boosting классификатор
        gb = GradientBoostingClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            learning_rate=0.1,
            max_depth=3
        )
        
        # Создаем XGBoost классификатор
        xgb = xgb.XGBClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            learning_rate=0.1,
            max_depth=3,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        # Создаем CatBoost классификатор
        catboost = CatBoostClassifier(
            iterations=n_estimators,
            random_seed=random_state,
            verbose=False,
            depth=3,
            learning_rate=0.1
        )
        
        return rf, gb, xgb, catboost

    def save_models(self, 
                   rf: RandomForestClassifier, 
                   gb: GradientBoostingClassifier, 
                   xgb_model: xgb.XGBClassifier, 
                   catboost: CatBoostClassifier):
        try:
            # Проверяем существование папки
            if not os.path.exists(PRO_PATH):
                os.makedirs(PRO_PATH)
            
            # Сохраняем модели
            dump(rf, RF_MODEL_PATH)
            dump(gb, GB_MODEL_PATH)
            dump(xgb_model, XGB_MODEL_PATH)
            catboost.save_model(CATBOOST_MODEL_PATH)
            print("Модели успешно сохранены")
            
        except Exception as e:
            print(f"Ошибка при сохранении моделей: {str(e)}")

    def load_models(self) -> Dict[str, object]:
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
            print(f"Ошибка: файл не найден - {str(e)}")
            return {}

    def load_scaler(self) -> StandardScaler:
        try:
            return load(SCALER_PATH)
        except FileNotFoundError:
            print(f"Ошибка: файл скалера не найден по пути {SCALER_PATH}")
            return None

    def save_scaler(self, scaler: StandardScaler):
        try:
            if not os.path.exists(PRO_PATH):
                os.makedirs(PRO_PATH)
            
            dump(scaler, SCALER_PATH)
            print("Скалер успешно сохранен")
        
        except Exception as e:
            print(f"Ошибка при сохранении скалера: {str(e)}")

# Функция проверки существования всех моделей
def check_models_exist():
    models_paths = [
        RF_MODEL_PATH,
        GB_MODEL_PATH,
        XGB_MODEL_PATH,
        CATBOOST_MODEL_PATH,
        SCALER_PATH
    ]
    
    for path in models_paths:
        if not os.path.exists(path):
            print(f"Предупреждение: файл {path} не найден")
            return False
    return True

if __name__ == "__main__":
    manager = ModelManager()
    
    # Пример использования
    try:
        # Проверка наличия моделей
        if not check_models_exist():
            print("Некоторые модели отсутствуют. Пожалуйста, проверьте пути сохранения.")
        else:
            print("Все модели найдены. Система готова к работе.")
            
            # Загрузка моделей
            models = manager.load_models()
            scaler = manager.load_scaler()
            
            print("Модели и скалер успешно загружены")
            
    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")

