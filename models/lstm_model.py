import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, LSTM, Dropout, Dense
from utils import compile_model

def build_lstm_model(num_layers: int = 2, units: int = 256, dropout_rate: float = 0.1, input_shape=(100, 20)):
    """
    Constructs an LSTM model with a variable number of layers (2, 3, 4, or 5) 
    using initial hyperparameter values of 256 units and 0.1 dropout rate 
    to reproduce Table 7 performance comparisons.
    
    Parameters:
    - num_layers: Number of LSTM layers (2, 3, 4, or 5)
    - units: Number of memory cells per LSTM layer (initial default: 256)
    - dropout_rate: Dropout rate applied after each LSTM layer (initial default: 0.1)
    - input_shape: Tuple representing sequence length and feature count (default: 100 timesteps, 20 features)
    
    Returns:
    - A compiled Keras Sequential model ready for training.
    """
    model = Sequential()
    
    # Define the input layer shape matching the sliding window sequence (Batch, 100, 20)
    model.add(Input(shape=input_shape))
    
    # Dynamically stack LSTM layers based on the specified layer count
    for i in range(num_layers):
        # All layers except the final one must return sequences so they can feed 
        # into the subsequent LSTM layer. The final layer returns a single vector.
        return_seq = (i < num_layers - 1)
        
        model.add(LSTM(units=units, return_sequences=return_seq))
        model.add(Dropout(rate=dropout_rate))
        
    # Dense output layer with 1 unit and a linear activation function 
    # to predict the continuous target stock price (next day's Adjusted Close)
    model.add(Dense(units=1, activation='linear'))
    
    # Compile the model using the shared configuration from utils.py (Adam optimizer, lr=0.001, MSE loss)
    model = compile_model(model, learning_rate=0.001)
    
    return model

if __name__ == "__main__":
    # Test building different layer configurations to replicate Table 7 testing phase
    for n in [2, 3, 4, 5]:
        print(f"\n--- LSTM Model Architecture: {n} Layers ---")
        test_model = build_lstm_model(num_layers=n)
        test_model.summary()