import os, sys
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.ensemble import RandomForestRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')
    model_set = {
        "Random Forest": {'regressor': RandomForestRegressor(), 'param':{'n_estimators': [8,16,32,64,128,256]}},
        "Decision Tree": {'regressor': DecisionTreeRegressor(), 'param':{
                    'min_weight_fraction_leaf':[0.001,0.01,0.1,0.3,0.5],
                    'min_impurity_decrease':[0.01,0.1,0.5,1]
                }},
        "Linear Regression": {'regressor': LinearRegression(), 'param':{}},
        "XGBoost": {'regressor': XGBRegressor(),'param':{'learning_rate':[0.001,0.01,0.05,0.1],'n_estimators':[8,16,32,64,128,256]}}
    }

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            model_set = self.model_trainer_config.model_set

            model_report: dict = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test= y_test, model_set=model_set)
            best_model_name, best_model_score = max(model_report.items(), key= lambda x: x[1])

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info("Best model found")

            save_object(model_set[best_model_name]['regressor'], file_path=self.model_trainer_config.trained_model_file_path)
            return best_model_score
        
        except Exception as e:
            raise CustomException(e, sys)