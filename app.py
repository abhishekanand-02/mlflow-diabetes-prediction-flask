from flask import Flask, render_template, request, jsonify
import os
import sys
import pandas as pd
from src.exception import customexception
from src.logger import logging
from src.pipelines.prediction_pipeline import PredictPipeline, CustomData

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get the data from the form input
            preg = float(request.form['Pregnancies'])
            glucose = float(request.form['Glucose'])
            blood_pressure = float(request.form['BloodPressure'])
            skin_thickness = float(request.form['SkinThickness'])
            insulin = float(request.form['Insulin'])
            bmi = float(request.form['BMI'])
            diabetes_pedigree_function = float(request.form['DiabetesPedigreeFunction'])
            age = float(request.form['Age'])

            # Create an instance of CustomData with the user's input
            custom_data = CustomData(preg, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age)

            # Convert the custom data into a DataFrame
            user_data_df = custom_data.get_data_as_dataframe()

            # Use PredictPipeline to make predictions
            prediction_pipeline = PredictPipeline()
            prediction = prediction_pipeline.predict(user_data_df)

            # Return prediction result
            result = (
                "This person is at risk of diabetes. 😞 Please take precautions and consider scheduling a checkup with your healthcare provider."
                if prediction[0] == 1
                else "This person is safe. Yay! 😄 Keep up the healthy lifestyle and maintain your good habits!"
            )


            return render_template('predict.html', prediction=result)

        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")
            return jsonify({"error": "An error occurred during prediction"}), 500
    else:
        return render_template('predict.html')


if __name__ == "__main__":
    app.run(debug=True)

