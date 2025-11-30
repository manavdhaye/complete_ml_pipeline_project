from src.exception import Custom_Exception
from src.logger import logging
from dataclasses import dataclass
from src.utils import save_object,evaluate_model
import os 
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,ExtraTreesRegressor
from sklearn.ensemble import BaggingRegressor

@dataclass
class ModelTrainerConfig:
    trained_model_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("split training and testing data")
            x_train,y_train,x_test,y_test=(train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1])
            models = {
            "Linear Regression": LinearRegression(),
            "Lasso": Lasso(alpha=0.001),
            "Ridge": Ridge(alpha=10),
            "K-Neighbors Regressor": KNeighborsRegressor(n_neighbors=4),
            "Decision Tree": DecisionTreeRegressor(max_depth=8),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=200,max_features="sqrt",max_depth=20,min_samples_leaf=1,min_samples_split=2),
            "ExtraTreesRegressor": ExtraTreesRegressor(n_estimators=100,max_features="sqrt",max_depth=20,min_samples_leaf=1,min_samples_split=2),
            "AdaBoost Regressor": AdaBoostRegressor(n_estimators=15,learning_rate=1.0),
            "GradientBoostingRegressor":GradientBoostingRegressor(n_estimators=500)
            }
            model_report:dict=evaluate_model(x_train,y_train,x_test,y_test,models)
            best_model_score=max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise Custom_Exception("no best modal found")
            logging.info("best modal found both traning and testing data")

            save_object(self.model_trainer_config.trained_model_path,best_model)

            predicted=best_model.predict(x_test)
            re_square=r2_score(y_test,predicted)
            return re_square

        except Exception as e:
            raise Custom_Exception(e,sys)



