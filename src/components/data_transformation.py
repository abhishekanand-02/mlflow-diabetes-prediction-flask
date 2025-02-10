import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import customexception
import os
import sys

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.utils.utils import save_object

class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation(self):
        try:
            logging.info('Data Transformation initiated')

            # Define numerical columns for diabetes dataset
            numerical_columns = [
                'Pregnancies', 'Glucose', 'BloodPressure', 
                'SkinThickness', 'Insulin', 'BMI', 
                'DiabetesPedigreeFunction', 'Age'
            ]

            logging.info('Pipeline Initiated')

            
            numerical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')), 
                    ('scaler', StandardScaler()) 
                ]
            )

            preprocessor = ColumnTransformer([
                ('numerical_pipeline', numerical_pipeline, numerical_columns)
            ])

            return preprocessor

        except Exception as e:
            logging.error(f"Exception occurred in get_data_transformation: {str(e)}")
            raise customexception(e, sys)

    def initialize_data_transformation(self, train_path, test_path):
        try:
            # Load train and test datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Successfully loaded train and test data.")
            
            # Get the data transformation pipeline
            preprocessing_pipeline = self.get_data_transformation()

            target_column = 'Outcome' 
            columns_to_drop = ['Outcome'] 

            X_train = train_df.drop(columns=columns_to_drop, axis=1) 
            y_train = train_df[target_column] 

            X_test = test_df.drop(columns=columns_to_drop, axis=1)  
            y_test = test_df[target_column]  

            X_train_transformed = preprocessing_pipeline.fit_transform(X_train)
            X_test_transformed = preprocessing_pipeline.transform(X_test)

            logging.info("Preprocessing completed on training and testing datasets.")

            training_data = np.c_[X_train_transformed, np.array(y_train)]
            testing_data = np.c_[X_test_transformed, np.array(y_test)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_pipeline
            )

            logging.info("Preprocessing object saved successfully.")

            return training_data, testing_data

        except Exception as e:
            logging.error(f"Exception occurred in initialize_data_transformation: {str(e)}")
            raise customexception(e, sys)


if __name__ == "__main__":
    train_data_path = 'artifacts/train.csv'
    test_data_path = 'artifacts/test.csv'

    data_transformation_instance = DataTransformation()
    train_data, test_data = data_transformation_instance.initialize_data_transformation(train_data_path, test_data_path)
