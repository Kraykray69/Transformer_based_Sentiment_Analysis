# Deployment Instructions

Follow these steps to set up and run the continuous training and deployment pipeline for the text classification model:

1. Environment Setup:
   - Ensure you have Python 3.7+ installed
   - Create a virtual environment:
         python -m venv venv
         source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
     
   ## - Install required packages:
         !pip install transformers torch pandas scikit-learn mlflow flask tqdm

## 2. Data Preparation:
   - Download the IMDB Dataset CSV file and place it in the project root directory(Data folder).

3. MLflow Setup:
   - Start the MLflow tracking server:
         mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow-artifacts

## 4. Go to the directory
    
## 5. Running the Pipeline:
   - Execute the main script:
         python main.py
   - This will train the initial model, log metrics and artifacts to MLflow, and save the model.

## 6. Starting the Inference Server:
   - Run the Flask app for live inference:
         python inference.py
   - The server will start on `http://localhost:5000`

7. Making Predictions:
   - For live inference, send a POST request to `http://localhost:5000/predict` with JSON data:
         curl -X POST -H "Content-Type: application/json" -d '{"text":"This movie was great!"}' http://localhost:5000/predict
   - For batch inference, use the `batch_inference` function in `Inference.py`

## 8. Monitoring and Retraining:
   - The `monitor_performance` function in `performance_monitoring.py` checks for model degradation.
   - If triggered, the `retrain_model` function in `retraining.py` will retrain the model.

## 9. Viewing MLflow Results:
   - Access the MLflow UI at `http://localhost:5000` to view experiment results and model versions.