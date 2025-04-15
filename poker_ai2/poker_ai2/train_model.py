# train_model.py

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder  # Added import
import joblib  # Added import

def load_and_preprocess_data(filename='poker_data.csv'):
    """
    Load data from CSV and preprocess it for training.

    :param filename: Path to the CSV file containing game data.
    :return: Tuple of (X_train, X_test, y_train, y_test, encoder)
    """
    # Load data
    data = pd.read_csv(filename)
    print(f"Loaded data with shape: {data.shape}")

    # Check for missing values
    if data.isnull().sum().any():
        print("Data contains missing values. Dropping missing entries.")
        data = data.dropna()

    # Separate features and labels
    X = data[['hand_rank', 'player_chips', 'current_bet', 'min_bet', 'pot', 'num_community_cards']].values
    y = data['action'].values

    # Verify action classes
    unique_actions = np.unique(y)
    print(f"Unique action classes in data: {unique_actions}")
    if unique_actions.max() >= 5:
        print("Adjusting num_classes to accommodate all action classes.")
        num_classes = unique_actions.max() + 1
    else:
        num_classes = 5  # As per action_mapping

    # One-hot encode the labels
    y_encoded = to_categorical(y, num_classes=num_classes)
    print(f"Features shape: {X.shape}, Labels shape: {y_encoded.shape}")

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")

    # Initialize and fit the encoder
    encoder = OneHotEncoder(sparse_output=False)  # Updated parameter
    encoder.fit(np.array(["Check", "Call", "Raise", "Fold", "All-In"]).reshape(-1, 1))
    print("Encoder fitted.")

    return X_train, X_test, y_train, y_test, encoder

def create_model(input_size, output_size):
    """
    Define and compile the neural network model.

    :param input_size: Number of input features.
    :param output_size: Number of output classes.
    :return: Compiled Keras model.
    """
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_size,)),
        Dense(32, activation='relu'),
        Dense(output_size, activation='softmax')  # Output layer for classification
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(X_train, X_test, y_train, y_test, input_size, output_size, epochs=20, batch_size=32):
    """
    Train the neural network model.

    :param X_train: Training features.
    :param X_test: Testing features.
    :param y_train: Training labels.
    :param y_test: Testing labels.
    :param input_size: Number of input features.
    :param output_size: Number of output classes.
    :param epochs: Number of training epochs.
    :param batch_size: Training batch size.
    :return: Trained Keras model and training history.
    """
    model = create_model(input_size, output_size)
    print("Starting model training...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=1
    )
    print("Model training completed.")
    return model, history

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test set.

    :param model: Trained Keras model.
    :param X_test: Testing features.
    :param y_test: Testing labels.
    """
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

def save_model(model, filename='poker_ai_model.keras'):
    """
    Save the trained model to disk in Keras format.

    :param model: Trained Keras model.
    :param filename: Filename to save the model.
    """
    model.save(filename)
    print(f"Model saved to {filename}")

def save_encoder(encoder, filename='encoder.joblib'):
    """
    Save the fitted OneHotEncoder to disk.

    :param encoder: Fitted OneHotEncoder instance.
    :param filename: Filename to save the encoder.
    """
    joblib.dump(encoder, filename)
    print(f"Encoder saved to {filename}")

if __name__ == "__main__":
    # Step 1: Load and preprocess data
    X_train, X_test, y_train, y_test, encoder = load_and_preprocess_data('poker_data2.csv')

    # Step 2: Define input and output sizes
    input_size = X_train.shape[1]  # 6 features
    output_size = y_train.shape[1]  # 5 actions

    # Step 3: Train the model
    model, history = train_model(
        X_train, X_test, y_train, y_test,
        input_size, output_size,
        epochs=20,  # Adjust epochs as needed
        batch_size=32  # Adjust batch size as needed
    )

    # Step 4: Evaluate the model
    evaluate_model(model, X_test, y_test)

    # Step 5: Save the trained model
    save_model(model, 'poker_ai_model.keras')

    # Step 6: Save the encoder
    save_encoder(encoder, 'encoder.joblib')
