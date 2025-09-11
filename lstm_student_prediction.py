# -*- coding: utf-8 -*-
"""
LSTM for Student Performance (Time-Series)
Cleaned version for deployment on Render.
"""

import zipfile
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
SEQUENCE_LENGTH = 3  # number of previous questions
DATA_FILE = "dataset/student-por.csv"  # path relative to project folder

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

    # Create binary target based on G3 (final grade)
    df['target'] = (df['G3'] >= 10).astype(int)
    df = df.drop(columns=['G3'])

    # Encode categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Create sequences
    X, y = create_sequences_for_next_step(df, sequence_length)

    if len(X) == 0:
        raise ValueError("Not enough data to create sequences. Reduce SEQUENCE_LENGTH.")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"✅ Training sequences: {X_train.shape[0]}")
    print(f"✅ Testing sequences: {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=False),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# --- Main Execution ---
if __name__ == "__main__":
    try:
        X_train, X_test, y_train, y_test = preprocess_data(DATA_FILE)

        model = build_lstm_model((X_train.shape[1], X_train.shape[2]))

        print("\n🚀 Training model...")
        model.fit(
            X_train, y_train, 
            epochs=50, 
            batch_size=32, 
            validation_data=(X_test, y_test), 
            verbose=1
        )
        print("🎉 Training complete.")

        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"\n📊 Model Evaluation:")
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy * 100:.2f}%")

        # Predict next answer for a sample student
        sample_sequence = X_test[-1].reshape(1, X_test.shape[1], X_test.shape[2])
        prediction = model.predict(sample_sequence, verbose=0)[0][0]
        actual_label = y_test[-1]

        print(f"\n🔮 Next Question Prediction Example:")
        print(f"Predicted Probability of Correct Answer: {prediction:.4f}")
        print(f"Prediction: {'Correct ✅' if prediction > 0.5 else 'Wrong ❌'}")
        print(f"Actual: {'Correct ✅' if actual_label == 1 else 'Wrong ❌'}")

    except Exception as e:
        print(f"❌ Error: {e}")

