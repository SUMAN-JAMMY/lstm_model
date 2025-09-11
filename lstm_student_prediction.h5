# lstm_train.py
# Train LSTM on student performance data and save model for deployment

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# --- Configuration ---
SEQUENCE_LENGTH = 3
DATA_FILE = "student-por.csv"  # Make sure this CSV is in your repo
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model_prediction.h5")

# --- Preprocessing Functions ---
def create_sequences_for_next_step(df_student, sequence_length):
    sequences, targets = [], []
    for i in range(len(df_student) - sequence_length):
        seq = df_student.iloc[i:i+sequence_length].drop(columns=['target']).values
        next_answer = df_student.iloc[i + sequence_length]['target']
        sequences.append(seq)
        targets.append(next_answer)
    return np.array(sequences), np.array(targets)

def preprocess_data(file_path, sequence_length=SEQUENCE_LENGTH):
    print("📥 Loading dataset...")
    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError("Dataset is empty!")

    # Convert final grade (G3) into binary target (pass/fail)
    df['target'] = (df['G3'] >= 10).astype(int)
    df = df.drop(columns=['G3'])

    # Encode categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    X, y = create_sequences_for_next_step(df, sequence_length)

    if len(X) == 0:
        raise ValueError("Not enough data to create sequences.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"✅ Training samples: {X_train.shape[0]}")
    print(f"✅ Testing samples: {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=False),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# --- Main Training ---
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data(DATA_FILE)

    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))

    print("\n🚀 Training model...")
    model.fit(X_train, y_train, epochs=20, batch_size=32,
              validation_data=(X_test, y_test), verbose=1)
    print("🎉 Training complete.")

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"📊 Test Accuracy: {acc*100:.2f}%")

    # Save model for deployment
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")
