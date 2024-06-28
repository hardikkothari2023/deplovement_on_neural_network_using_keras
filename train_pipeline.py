import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import MeanSquaredError
from sklearn.preprocessing import StandardScaler
from src.preprocessing.data_management import load_dataset, save_model

def preprocess_data(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def build_model(input_dim):
    model = Sequential()
    model.add(Dense(4, input_dim=input_dim, activation='relu'))
    model.add(Dense(2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model

def run_training(tol, learning_rate, max_epochs):
    training_data = load_dataset("train.csv")
    X_train, Y_train = training_data.iloc[:, 0:2], training_data.iloc[:, 2]
    X_train, Y_train = preprocess_data(X_train, Y_train)
    
    # Build model
    model = build_model(input_dim=X_train.shape[1])
    
    # Compile model
    model.compile(optimizer=SGD(learning_rate=learning_rate), loss=MeanSquaredError())

    # Train model
    history = model.fit(X_train, Y_train, epochs=max_epochs, batch_size=2, verbose=1)
    
    # Save model
    save_model(model, "two_input_xor_nn.pkl")
    
    # Print loss per epoch
    for epoch, loss in enumerate(history.history['loss']):
        print(f"Epoch # {epoch + 1}, Loss = {loss}")
        
        if loss < tol:
            break

if __name__ == "__main__":
    run_training(1e-20, 0.001, 5000)
