import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from mlflow.models import infer_signature
import os
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from urllib.parse import urlparse
from sklearn.preprocessing import LabelEncoder
import mlflow

# Set MLflow environment variables for DagsHub tracking
os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/negisahil300103/WaterQualityPrediction.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = "negisahil300103"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "f6bbde60dc93d2aaae8a8836cc7d76000576bc3a"

def train(data_path, model_path, random_state):
    """
    Train a Random Forest model, log metrics and artifacts to MLflow, and save the model locally.
    """
    # Load the dataset
    data = pd.read_csv(data_path)
    
    # Drop rows with missing target values
    data.dropna(subset=["Target"], inplace=True)
    
    # Split into features and target
    X = data.drop(columns=["Target"])
    y = data["Target"]
    
    # Encode categorical columns
    for column in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
    
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

    with mlflow.start_run():
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=random_state)
        signature = infer_signature(X_train, y_train)
        
        # Define hyperparameter distributions
        param_distributions = {
            'n_estimators': np.arange(10, 30, 5),
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 4, 6],
            'min_samples_leaf': [1, 2],
            'bootstrap': [True, False]
        }
        
        # Create and run RandomizedSearchCV
        rf = RandomForestClassifier(random_state=42)
        random_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_distributions,
            n_iter=5,
            cv=2,
            scoring='accuracy',
            random_state=42
        )
        random_search.fit(X_train, y_train)
        
        # Get the best model
        best_model = random_search.best_estimator_
        
        # Predict and evaluate the model
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")
        
        # Log metrics and best parameters to MLflow
        mlflow.log_metric("accuracy", accuracy)
        for param, value in random_search.best_params_.items():
            mlflow.log_param(f"best_{param}", value)
        
        labels = [0, 1]  # Change according to your actual labels (0 = Unsafe, 1 = Safe)
        
        # Log confusion matrix and classification report
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        cr = classification_report(y_test, y_pred, labels=labels)
        mlflow.log_text(str(cm), "confusion_matrix.txt")
        mlflow.log_text(cr, "classification_report.txt")
        
        # Log the model
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        if tracking_url_type_store != 'file':
            mlflow.sklearn.log_model(best_model, "model", registered_model_name="Best Model")
        else:
            mlflow.sklearn.log_model(best_model, "model", signature=signature)
        
        # Save the model locally
        model_dir = os.path.dirname(model_path) or "."
        os.makedirs(model_dir, exist_ok=True)
        pickle.dump(best_model, open(model_path, 'wb'))
        print(f"Model saved to {model_path}")

if __name__ == "__main__":
    # Load parameters from params.yaml
    params = yaml.safe_load(open("params.yaml"))["train"]
    train(params['data'], params['model'], params['random_state'])
