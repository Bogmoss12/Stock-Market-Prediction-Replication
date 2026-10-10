import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, GRU, Dropout, Dense
from utils import compile_model

# GRU model with initial hyperparameter of 256 and 0.1 dropout rate and starting with 2 layers

def build_gru_model(num_layers: int = 2, units: int = 256, dropout_rate: float = 0.1, input_shape = (100, 20)):
    model = Sequential()

    # Define input layer shape with sliding window sequence
    model.add(Input(shape=input_shape))

    # Stack GRU layers
    for i in range(num_layers):
        return_seq = (i < num_layers - 1)

        model.add(GRU(units=units, return_sequences=return_seq))
        model.add(Dropout(rate=dropout_rate))

    # Dense output layer with linear activation function
    model.add(Dense(units=1, activation='linear'))

    model = compile_model(model, learning_rate=0.001)

    return model

# Test building different layer configs
if __name__ == "__main__":
    for n in [2, 3, 4]:
        print(f"\n--- GRU Model Architecture: {n} Layers ---")
        test_model = build_gru_model(num_layers=n)
        test_model.summary()