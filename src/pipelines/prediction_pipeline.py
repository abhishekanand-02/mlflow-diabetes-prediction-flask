import os
import sys
import pandas as pd
from src.exception import customexception
from src.logger import logging
from src.utils.utils import load_object  # Ensure this is defined somewhere

class PredictPipeline:
    def __init__(self):
        logging.info("Initializing the prediction pipeline object...")

    def predict(self, features, target=None):
        try:
            # Load preprocessor and model from the artifacts folder
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model_path = os.path.join("artifacts", "model.pkl")

            logging.info("Loading preprocessor and model from the artifacts...")
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            # Apply preprocessor transformation to the input data
            logging.info("Applying preprocessing to the input data...")
            scaled_features = preprocessor.transform(features)  # Preprocessing the input features
            predictions = model.predict(scaled_features)  # Predicting with the trained model

            # Handle model score calculation if target is provided and has more than one sample
            if target is not None and len(target) > 1:
                score = model.score(scaled_features, target)
                logging.info(f"Model Score (R^2): {score}")
            else:
                logging.info("Model score calculation skipped for single sample prediction.")

            # Log actual vs predicted values
            logging.info("Actual vs Predicted values:")
            if target is not None:
                for actual, predicted in zip(target, predictions):
                    logging.info(f"Actual: {actual}, Predicted: {predicted}")
            else:
                for predicted in predictions:
                    logging.info(f"Predicted: {predicted}")

            logging.info("Prediction completed successfully.")
            return predictions

        except Exception as e:
            logging.error(f"Exception occurred during prediction: {str(e)}")
            raise customexception(e, sys)

class CustomData:
    def __init__(self,
                 preg: float,
                 glucose: float,
                 blood_pressure: float,
                 skin_thickness: float,
                 insulin: float,
                 bmi: float,
                 diabetes_pedigree_function: float,
                 age: float):
        """
        Initializes the custom data class with input features for diabetes prediction.
        :param preg: Number of pregnancies
        :param glucose: Plasma glucose concentration
        :param blood_pressure: Diastolic blood pressure (mm Hg)
        :param skin_thickness: Triceps skinfold thickness (mm)
        :param insulin: 2-Hour serum insulin (mu U/ml)
        :param bmi: Body mass index (weight in kg / (height in m)^2)
        :param diabetes_pedigree_function: Diabetes pedigree function
        :param age: Age (years)
        """
        self.preg = preg
        self.glucose = glucose
        self.blood_pressure = blood_pressure
        self.skin_thickness = skin_thickness
        self.insulin = insulin
        self.bmi = bmi
        self.diabetes_pedigree_function = diabetes_pedigree_function
        self.age = age

    def get_data_as_dataframe(self):
        try:
            # Prepare input data dictionary for diabetes dataset
            custom_data_input_dict = {
                'Pregnancies': [self.preg],
                'Glucose': [self.glucose],
                'BloodPressure': [self.blood_pressure],
                'SkinThickness': [self.skin_thickness],
                'Insulin': [self.insulin],
                'BMI': [self.bmi],
                'DiabetesPedigreeFunction': [self.diabetes_pedigree_function],
                'Age': [self.age]
            }
            # Convert to DataFrame
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe created successfully.')
            return df
        except Exception as e:
            logging.error('Exception occurred while creating the dataframe.')
            raise customexception(e, sys)

# Example Usage:
if __name__ == "__main__":
    # Create an instance of the custom data with appropriate values
    custom_data = CustomData(
        preg=6,               # Number of pregnancies
        glucose=148,          # Plasma glucose concentration
        blood_pressure=72,    # Diastolic blood pressure
        skin_thickness=35,    # Triceps skinfold thickness
        insulin=0,            # 2-Hour serum insulin
        bmi=33.6,             # Body mass index
        diabetes_pedigree_function=0.627,  # Diabetes pedigree function
        age=50                # Age
    )
    
    # Convert the custom data into a DataFrame
    input_data = custom_data.get_data_as_dataframe()

    # Initialize the prediction pipeline and run prediction
    predict_pipeline = PredictPipeline()
    predictions = predict_pipeline.predict(input_data)
    
