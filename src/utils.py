import os,sys
from src.exception import CustomException
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from dataclasses import dataclass


def save_object(obj, file_path):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as fp:
            dill.dump(obj, fp)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(X_train, y_train, X_test, y_test, model_set):
    try:
        report = {}
        for model_name in model_set:
            model = model_set[model_name]['regressor']
            param = model_set[model_name]['param']
            gs = GridSearchCV(model, param, cv = 3)
            gs.fit(X_train, y_train)
            model.set_params(**gs.best_params_)

            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_model_score = r2_score(y_train_pred, y_train)
            test_model_score = r2_score(y_test_pred, y_test)
            report[model_name] = test_model_score
        return report
    
    except Exception as e:
        raise CustomException(e, sys)



def load_object(file_path):
    try:
        with open(file_path, 'rb') as fp:
            return dill.load(fp)
    except Exception as e:
        raise CustomException(e, sys)
