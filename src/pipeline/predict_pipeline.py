from src.exception import Custom_Exception
from src.logger import logging
from src.utils import load_object
import os
import sys
import pandas as pd
import math
import numpy as np

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(model_path)
            print("After Loading")
            logging.info("model loaded in predict function")
            preprocessor=load_object(preprocessor_path)
            data_scaled=preprocessor.transform(features)
            result=model.predict(data_scaled)
            output=math.floor(np.exp(result[0]))
            logging.info("result display of prediction")
            return output
        except Exception as e:
            raise Custom_Exception(e,sys)


class CustomData:
    def __init__(self,company:str,typename:str,ram:int,weight:float,touchscrren:int,ips:int,ppi:float,cpu:str,gpu:str,os:str,HDD:int,SDD:int):
        self.company=company
        self.typename=typename
        self.ram=ram
        self.weight=weight
        self.touchscrren=touchscrren
        self.ips=ips
        self.ppi=ppi
        self.cpu=cpu
        self.gpu=gpu
        self.os=os
        self.HDD=HDD
        self.SDD=SDD

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict={
            "Company": [self.company],
            "TypeName":[self.typename],
            "Ram":[self.ram],
            "Weight":[self.weight],
            "Touchscreen":[self.touchscrren],
            "IPS":[self.ips],
            "ppi":[self.ppi],
            "CPU_brand":[self.cpu],
            "Gpu_brand":[self.gpu],
            "os":[self.os],
            "HDD":[self.HDD],
            "SDD":[self.SDD],
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise Custom_Exception(e,sys)
    
