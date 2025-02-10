# mlflow-diabetes-prediction-flask
A Machine Learning Prediction App to predict diabetes risk using a Flask web app.

---

# Project Overview
This project provides a machine learning-based diabetes prediction application built using **Flask**. It incorporates **MLflow** for managing the ML lifecycle, including experiments, model training, and deployment. It also integrates **DVC** (Data Version Control) to handle and version large datasets.

---

# Table of Contents
1. [Project Setup](#project-setup)
2. [Running the Application](#running-the-application)
3. [MLflow Integration](#mlflow-integration)
4. [DVC Integration](#dvc-integration)
5. [Flask Web App](#flask-web-app)
6. [Components and Pipelines](#components-and-pipelines)
7. [Conclusion](#conclusion)

---

# Project Setup

## **1. Initialize Environment**
First, clone the repository and set up the virtual environment.

```bash
git clone <your-repository-url>
cd mlflow-diabetes-prediction-flask
```


Create a virtual environment:

```bash
python3 -m venv virtual_env
source virtual_env/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## **2. Make init-setup.sh Executable**
The init-setup.sh script helps in setting up the project and required services. Make sure it's executable:

```bash
chmod +x init-setup.sh
```

Run the script to initialize the setup:

```bash
./init-setup.sh
```
# **Running the Application**

## 1. Start the Flask Web App
After setting up your environment and dependencies, you can run the Flask app:
```bash
python app.py
```
The application will be available at http://localhost:5000. You can navigate to this URL on your browser and interact with the prediction functionality.

# **MLflow Integration**
## 1. MLflow Setup
MLflow is used to manage the ML model lifecycle, including tracking experiments, managing models, and deployment.

To track experiments using MLflow:

The experiment scripts can be found in the /experiment directory.
experiments.ipynb helps you in running experiments and tracking them using MLflow's tracking features.

## 2. Running the MLflow Server
To track experiments in real-time, you can run the MLflow server:

```bash
mlflow ui
```

This will start a server at http://localhost:5000, where you can track and compare different experiments.

# **DVC Integration**
DVC is used to version control large datasets. We have integrated DVC for managing the diabetes dataset.

## 1. Initialize DVC in the Project
In the project root directory, initialize DVC:

```bash
dvc init
```

## 2. Track the Dataset Using DVC
The diabetes dataset is located in csv_store/diabetes.csv. To add the dataset to DVC:

```bash

dvc add csv_store/diabetes.csv
```

This will create the necessary .dvc file and update .gitignore to ensure that large data files are ignored by Git and managed by DVC.

## 3. Stage and Commit Changes to Git
Track the changes and commit them to Git:

```bash

git add csv_store/.gitignore csv_store/diabetes.csv.dvc
git commit -m "Track diabetes.csv using DVC"
git push  # Push changes to the remote repository
```

## 4. Pushing to DVC Remote Storage (Optional)
If you have a remote DVC storage (like AWS S3, GCP, etc.), push the data to it:

```bash

dvc push
```

# **Flask Web App**
The Flask web app provides a simple interface for users to input their data and get predictions on whether they are at risk of diabetes.

## 1. Flask Setup
The app.py file contains the logic for the web server, and the /predict endpoint allows users to submit their data for predictions.

```bash
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Handles form submission and prediction logic
```

## 2. App Routes
/predict: Where users can input their data for prediction.
/: The home page where users can learn more about the app.

# Components and Pipelines
The project includes the following main components:

### 1. Data Ingestion (data_ingestion.py)
Responsible for loading and preparing the dataset.

### 2. Data Transformation (data_transformation.py)
Handles the preprocessing and cleaning of the data for model training.

### 3. Model Trainer (model_trainer.py)
Trains the diabetes prediction model using machine learning algorithms.

### 4. Model Evaluation (model_evaluation.py)
Evaluates the trained model’s performance using appropriate metrics (accuracy, F1-score, etc.).

# Conclusion
This project provides a complete pipeline for predicting diabetes risk using machine learning. By using Flask for the web app, MLflow for experiment tracking and model management, and DVC for versioning large datasets, this system allows efficient experimentation and deployment for healthcare predictions.

