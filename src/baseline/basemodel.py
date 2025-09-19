import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

REGRESSORS = {
    'lr': LinearRegression,
    'ridge': Ridge,
    'lasso': Lasso,
    'elasticnet': ElasticNet,
    'randomforest': RandomForestRegressor,
    'xgb': XGBRegressor,
    'lgbm': LGBMRegressor,
    'catboost': CatBoostRegressor
}

def get_regressor(reg:str):
    regressor = reg.strip().lower()
    if regressor not in REGRESSORS:
        raise ValueError(f"Regressor '{reg}' is not supported. Choose from {list(REGRESSORS.keys())}.")
    return REGRESSORS[regressor]()
        

class BaseModel():
    def __init__(self, model:str, params:dict = {}):
        self.model = get_regressor(model)
        self.params = params

    def fit(self, X_train, y_train):
        self.model.set_params(**self.params)
        self.model.fit(X_train, y_train)
        return self.model
    
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, y_true, y_pred):
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        return rmse, mae
    
    def feature_importance(self, X):
        importances = self.model.feature_importances_
        feature_names = X.columns
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
        return importance_df
    
