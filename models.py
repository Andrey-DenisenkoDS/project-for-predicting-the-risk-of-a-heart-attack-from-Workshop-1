from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
from joblib import dump, load
import os
from sklearn.preprocessing import StandardScaler

MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

RF_MODEL_PATH = os.path.join(MODEL_DIR, 'rf_model.pkl')
GB_MODEL_PATH = os.path.join(MODEL_DIR, 'gb_model.pkl')
XGB_MODEL_PATH = os.path.join(MODEL_DIR, 'xgb_model.pkl')
CATBOOST_MODEL_PATH = os.path.join(MODEL_DIR, 'catboost_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')

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

    def load_scaler(self):
        return load(SCALER_PATH)

    def save_scaler(self, scaler):
        dump(scaler, SCALER_PATH)


