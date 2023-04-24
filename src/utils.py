import sys
import os
import pandas as pd
import numpy as np
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score


def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(x_train, y_train, x_test, y_test, models):
    try:
        report = {}
        # models_list = []
        # r2_list = []

        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(x_train, y_train)

            # make pred
            y_train_pred = (model.predict(x_train))
            y_test_pred = (model.predict(x_test))

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score
        return report

    except Exception as e:
        raise CustomException(e, sys)
