# model/train_model.py

import numpy as np
import tensorflow as tf
from utils.encoder import UCI_MOVES

# Load dataset
def load_data(npz_path):
    data = np.load(npz_path)
    X = data["X"]
    y = data["y"]
    return X, y

# Build a simple CNN model
def build_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    print("Loading dataset...")
    X, y = load_data("data/train_data.npz")
    print(f"Dataset shape: X={X.shape}, y={y.shape}")

    print("Building model...")
    model = build_model(input_shape=(8, 8, 12), num_classes=len(UCI_MOVES))

    print("Training...")
    history = model.fit(X, y, epochs=100, batch_size=64, validation_split=0.1)
    
    print("Saving model...")
    model.save("model/chess_model.keras")
    print("Model saved to model/chess_model.keras")
