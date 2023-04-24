import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import RandomizedSearchCV
# from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import warnings
import sys
import os
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj
from src.utils import evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info('Splitting train and test input data')

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]

            )

            models = {
                "linear reg": LinearRegression(),
                "lasso": Lasso(),
                'ridge': Ridge(),
                'k-neighbors': KNeighborsRegressor(),
                'decision tree': DecisionTreeRegressor(),
                'random_forest': RandomForestRegressor(),
                'XGboost reg': XGBRegressor(),
                # 'Catboost reg': CatBoostRegressor(),
                'adaboost regressor': AdaBoostRegressor()
            }

            models_report: dict = evaluate_model(
                x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, models=models)
            logging.info('metrics have been calculated')
            # to get best models= score from dict

            best_model_score = max(sorted(models_report.values()))
            # to get best model name
            best_model_name = list(models_report.keys())[list(
                models_report.values()).index(best_model_score)]
            logging.info('best model is selected')
            best_model = models[best_model_name]
            if best_model_score < .6:
                raise CustomException('no best model found')

            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted = best_model.predict(x_test)
            r2_square = r2_score(y_test, predicted)
            return (r2_square)

        except Exception as e:
            raise CustomException(e, sys)
