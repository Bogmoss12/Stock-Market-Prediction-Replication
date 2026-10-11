import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, SimpleRNN, Dropout, Dense
from utils import compile_model

def build_rnn_model(num_layers: int = 2, units: int = 256, dropout_rate: float = 0.1, input_shape=(100, 20)):
    """
    Constructs a Recurrent Neural Network (RNN) model using SimpleRNN layers 
    to serve as one of the baseline comparison architectures from the paper.
    
    Parameters:
    - num_layers: Number of SimpleRNN layers (typically evaluated alongside 2 to 5 layers)
    - units: Number of memory cells per RNN layer (default: 256 to maintain parity)
    - dropout_rate: Dropout rate applied after each RNN layer (default: 0.1)
    - input_shape: Tuple representing sequence length and feature count (default: 100 timesteps, 20 features)
    
    Returns:
    - A compiled Keras Sequential model ready for training.
    """
    model = Sequential()
    
    # Define the input layer shape matching the sliding window sequence (Batch, 100, 20)
    model.add(Input(shape=input_shape))
    
    # Dynamically stack SimpleRNN layers based on the specified layer count
    for i in range(num_layers):
        # All layers except the final one must return sequences to feed into the next RNN layer
        return_seq = (i < num_layers - 1)
        
        model.add(SimpleRNN(units=units, return_sequences=return_seq))
        model.add(Dropout(rate=dropout_rate))
        
    # Dense output layer with 1 unit and a linear activation function 
    # to predict the continuous target stock price (next day's Adjusted Close)
    model.add(Dense(units=1, activation='linear'))
    
    # Compile the model using the shared configuration from utils.py (Adam optimizer, lr=0.001, MSE loss)
    model = compile_model(model, learning_rate=0.001)
    
    return model

if __name__ == "__main__":
    # Quick sanity check / test of the RNN model build and summary display
    print("\n--- RNN Model Architecture Test ---")
    rnn_test_model = build_rnn_model(num_layers=2)
    rnn_test_model.summary()