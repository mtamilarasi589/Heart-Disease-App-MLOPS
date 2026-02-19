from flask import Flask, render_template, request, jsonify
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
MODEL_PATH = os.path.join(MODEL_DIR, "heart_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
DATA_PATH = os.path.join(BASE_DIR, "model", "heart.csv")

# Ensure saved_models directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------
# Step 1: Train model if not exists
# -------------------------
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    print("Training new model...")

    # Load dataset
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"CSV file not found at {DATA_PATH}")
    data = pd.read_csv(DATA_PATH)

    X = data.drop("target", axis=1)
    y = data["target"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save scaler
    joblib.dump(scaler, SCALER_PATH)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Build MLP neural network
    model = MLPClassifier(hidden_layer_sizes=(16, 8), activation='relu',
                          solver='adam', max_iter=500, random_state=42)

    # Train model
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, MODEL_PATH)
    print("Model trained and saved.")

else:
    print("Loading existing model and scaler...")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

# -------------------------
# Step 2: Routes
# -------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Convert input to numeric array
    features = np.array([
        float(data['age']), float(data['sex']), float(data['cp']), float(data['trestbps']),
        float(data['chol']), float(data['fbs']), float(data['restecg']), float(data['thalach']),
        float(data['exang']), float(data['oldpeak']), float(data['slope']), float(data['ca']),
        float(data['thal'])
    ]).reshape(1, -1)

    # Scale features
    features_scaled = scaler.transform(features)

    # Predict
    prediction = model.predict(features_scaled)[0]
    result = "High risk" if prediction == 1 else "Low risk"

    return jsonify({"prediction": result})

# -------------------------
# Step 3: Run App
# -------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
