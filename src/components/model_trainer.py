import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import customexception
import os
import sys
from src.utils.utils import save_object
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info('Splitting dependent and independent variables from train and test data.')

            # Splitting the features (X) and target (y) from the training and testing arrays
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],  # All columns except the last column as features
                train_array[:, -1],   # Last column as target
                test_array[:, :-1], 
                test_array[:, -1]    
            )

            # Logistic Regression model
            model = LogisticRegression(max_iter=1000)

            # Training the model
            model.fit(X_train, y_train)

            # Predict on the test set
            y_pred = model.predict(X_test)

            # Evaluate model using accuracy score
            accuracy = accuracy_score(y_test, y_pred)

            logging.info(f"Model Accuracy: {accuracy}")
            print(f"Model Accuracy: {accuracy}")

            # Save the trained model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )

            logging.info(f"Best Model Found: Logistic Regression with Accuracy: {accuracy}")
            print(f"Best Model Found: Logistic Regression with Accuracy: {accuracy}")

        except Exception as e:
            logging.error(f"Exception occurred during model training: {str(e)}")
            raise customexception(e, sys)

if __name__ == "__main__":
    # Load the train and test data from CSV files
    train_data_path = 'artifacts/train.csv'
    test_data_path = 'artifacts/test.csv'
    
    # Load CSV files using pandas
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    
    # Convert DataFrames to numpy arrays
    train_array = train_data.values
    test_array = test_data.values

    # Create an instance of the ModelTrainer class and initiate training
    trainer = ModelTrainer()
    trainer.initiate_model_training(train_array, test_array)
