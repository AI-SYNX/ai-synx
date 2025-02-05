# models/model.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import json

class SYNXModel:
    def __init__(self, model_config_path='models/model_config.json'):
        """
        Initializes the SYNX Model with configuration settings.
        Loads pre-trained model configurations from a JSON file.
        """
        self.model = LinearRegression()
        self.scaler = StandardScaler()

        # Load model config from JSON
        with open(model_config_path, 'r') as config_file:
            self.config = json.load(config_file)
        
        print("Model initialized with configuration: ", self.config)

    def preprocess_data(self, data):
        """
        Preprocess the data by scaling it.
        """
        print("Preprocessing data...")
        return self.scaler.fit_transform(data)

    def train(self, X_train, y_train):
        """
        Train the model on the provided training data.
        """
        X_train_scaled = self.preprocess_data(X_train)
        print("Training the model...")
        self.model.fit(X_train_scaled, y_train)
        print("Training completed.")

    def predict(self, X_test):
        """
        Predict using the trained model.
        """
        X_test_scaled = self.scaler.transform(X_test)
        predictions = self.model.predict(X_test_scaled)
        return predictions

    def save_model(self, model_path):
        """
        Save the trained model and scaler to disk for later use.
        """
        print(f"Saving model to {model_path}...")
        with open(model_path, 'wb') as model_file:
            pickle.dump(self.model, model_file)
        with open(model_path.replace('model', 'scaler'), 'wb') as scaler_file:
            pickle.dump(self.scaler, scaler_file)
        print("Model saved successfully.")

    def load_model(self, model_path, scaler_path):
        """
        Load a saved model and scaler.
        """
        print(f"Loading model from {model_path}...")
        with open(model_path, 'rb') as model_file:
            self.model = pickle.load(model_file)
        with open(scaler_path, 'rb') as scaler_file:
            self.scaler = pickle.load(scaler_file)
        print("Model loaded successfully.")

if __name__ == "__main__":
    # Example usage
    model = SYNXModel()

    # Load training data (this would normally come from your dataset)
    X_train = np.array([[1, 2], [3, 4], [5, 6]])
    y_train = np.array([1, 2, 3])

    model.train(X_train, y_train)
    predictions = model.predict(np.array([[7, 8]]))
    print("Predictions: ", predictions)
