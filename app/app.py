from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import joblib
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

app = Flask(__name__)

MODEL_PATH = "saved_models/heart_model.h5"
SCALER_PATH = "saved_models/scaler.pkl"
DATA_PATH = "model/heart.csv"  # Ensure heart.csv is in app/model/

# -------------------------
# Step 1: Train model if it doesn't exist
# -------------------------
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    print("Pre-trained model not found. Training model...")

    # Load dataset
    data = pd.read_csv(DATA_PATH)
    X = data.drop("target", axis=1)
    y = data["target"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save scaler
    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Build neural network
    model = Sequential([
        Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train model
    model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1, verbose=1)

    # Save model
    model.save(MODEL_PATH)
    print("Training complete. Model and scaler saved.")
else:
    # -------------------------
    # Step 2: Load pre-trained model and scaler
    # -------------------------
    print("Loading pre-trained model and scaler...")
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

# -------------------------
# Step 3: Routes
# -------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Convert input to numeric array
    features = np.array([float(data['age']), float(data['sex']), float(data['cp']), float(data['trestbps']),
                         float(data['chol']), float(data['fbs']), float(data['restecg']), float(data['thalach']),
                         float(data['exang']), float(data['oldpeak']), float(data['slope']), float(data['ca']),
                         float(data['thal'])]).reshape(1, -1)

    # Scale features
    features_scaled = scaler.transform(features)

    # Predict
    prediction = model.predict(features_scaled)[0][0]
    result = "High risk" if prediction > 0.5 else "Low risk"

    return jsonify({"prediction": result})

# -------------------------
# Step 4: Run App
# -------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
