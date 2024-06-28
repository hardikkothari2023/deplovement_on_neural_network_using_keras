import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load the model
model = load_model("two_input_xor_nn.pkl")

# Function to preprocess input data
def preprocess_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def predict_xor(input_data):
    # Preprocess the input data
    X = preprocess_data(np.array(input_data).reshape(1, -1))
    
    # Predict the output using the model
    prediction = model.predict(X)
    
    # Return the predicted value
    return prediction[0][0]

def calculate_accuracy():
    # Define the XOR inputs and expected outputs
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    expected_outputs = np.array([0, 1, 1, 0])
    
    # Preprocess the inputs
    inputs_scaled = preprocess_data(inputs)
    
    # Predict the outputs
    predictions = model.predict(inputs_scaled)
    
    # Convert predictions to binary outputs
    predictions = (predictions > 0.5).astype(int).flatten()
    
    # Calculate accuracy
    accuracy = np.mean(predictions == expected_outputs)
    return accuracy, predictions

if __name__ == "__main__":
    # Calculate accuracy
    accuracy, predictions = calculate_accuracy()
    
    # Print predictions and accuracy
    print("Predictions for XOR inputs [0, 0], [0, 1], [1, 0], [1, 1]:")
    print(predictions)
    print(f"Model accuracy: {accuracy * 100:.2f}%")
