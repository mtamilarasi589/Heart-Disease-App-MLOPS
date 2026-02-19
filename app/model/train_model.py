import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# -------------------------
# Step 1: Load Dataset
# -------------------------
data = pd.read_csv("heart.csv")  # Make sure heart.csv is in app/model/

# Features and target
X = data.drop("target", axis=1)
y = data["target"]

# -------------------------
# Step 2: Scale Features
# -------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create folder for saved models
os.makedirs("../saved_models", exist_ok=True)

# Save the scaler for later
joblib.dump(scaler, "../saved_models/scaler.pkl")

# -------------------------
# Step 3: Split Dataset
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -------------------------
# Step 4: Build Neural Network
# -------------------------
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# -------------------------
# Step 5: Train Model
# -------------------------
print("Training model...")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.1,
    verbose=1
)

# -------------------------
# Step 6: Evaluate Model
# -------------------------
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# -------------------------
# Step 7: Save Model
# -------------------------
model.save("../saved_models/heart_model.h5")
print("Model and scaler saved to 'saved_models/' folder")
