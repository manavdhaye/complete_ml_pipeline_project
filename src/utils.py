import numpy as np
import pandas as pd
from src.exception import Custom_Exception
import os
import sys
import dill
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
        
    except Exception as e:
        raise Custom_Exception(e,sys)
 
def evaluate_model(x_train,y_train,x_test,y_test,models):
    report={}
    for i in range(len(list(models))):
        model = list(models.values())[i]
        model.fit(x_train,y_train)

        y_train_pred=model.predict(x_train)
        y_test_pred = model.predict(x_test)

        # mae = mean_absolute_error(y_train, y_train_pred)
        # mse = mean_squared_error(y_train, y_train_pred)
        # rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

        train_r2_square = r2_score(y_train, y_train_pred)
        test_r2_square = r2_score(y_test, y_test_pred)

        report[list(models.keys())[i]]=test_r2_square
    return report

def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise Custom_Exception(e,sys)

