import os
import sys
from src.exception import Custom_Exception
from src.logger import logging
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")

class DataFormation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_trasnformer_objects(self):
        try:
            numerical_columns=["Ram","Weight","Touchscreen","IPS","ppi","HDD","SDD"]
            categorical_columns=["Company","TypeName","CPU_brand","Gpu_brand","os"]

            logging.info(f"categorical columns :{categorical_columns}")
            logging.info(f"numerical columns :{numerical_columns}")

            # step1=ColumnTransformer(transformers=[("col_tnf",OneHotEncoder(sparse_output=False,drop="first"),[0,1,7,8,9])],remainder="passthrough")
            # step2=RandomForestRegressor(n_estimators=200,max_features="sqrt",max_depth=20,min_samples_leaf=1,min_samples_split=2)
            # pipe=Pipeline([("step1",step1),("step2",step2)])
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median"))
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot", OneHotEncoder(handle_unknown="ignore", drop="first"))
                ]
            )
            logging.info("Categorical columns one hot encosing completing")


            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ],
                remainder="drop"
            )
            return preprocessor

        except Exception as e:
            raise Custom_Exception(e,sys)
    
    def initiate_data_transformer(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Read train and test data complete")
            logging.info("obtaning preprocessing object")
            preprocessing_object=self.get_data_trasnformer_objects()
            target_column_name="Price"
            # print("Train Columns:", train_df.columns.tolist())
            # print("Test Columns:", test_df.columns.tolist())
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=np.log(train_df[target_column_name])

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=np.log(test_df[target_column_name])

            logging.info("applying preprocessor object on tranig and test dataframe")
            input_feature_train_arr=preprocessing_object.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_object.transform(input_feature_test_df)

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            logging.info("Saved preprocessing object.")

            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,obj=preprocessing_object)
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise Custom_Exception(e,sys)
        

