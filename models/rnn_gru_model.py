import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, SimpleRNN, GRU, Dropout, Dense
from utils import compile_model 

# Stacks RNN then GRU layers
def build_rnn_gru_model(rnn_layers: int, gru_layers: int, stack_order: str = 'rnn_first', units: int = 256, dropout_rate: float = 0.1, input_shape=(100, 20)):
    model = Sequential()

    # Define input layer shape and total layers
    model.add(Input(shape=input_shape))
    total_layers = rnn_layers + gru_layers
    current_layer = 0
    
    if stack_order == 'rnn_first':
        # Stack the RNN layers
        for i in range(rnn_layers):
            current_layer += 1
            return_seq = (current_layer < total_layers)

            model.add(SimpleRNN(units=units, return_sequences=return_seq))
            model.add(Dropout(rate=dropout_rate))

        # Stack the GRU layers
        for i in range(gru_layers):
            current_layer += 1
            return_seq = (current_layer < total_layers)

            model.add(GRU(units=units, return_sequences=return_seq))
            model.add(Dropout(rate=dropout_rate))
            
    elif stack_order == 'gru_first':
        # Stack the GRU layers
        for i in range(gru_layers):
            current_layer += 1
            return_seq = (current_layer < total_layers)

            model.add(GRU(units=units, return_sequences=return_seq))
            model.add(Dropout(rate=dropout_rate))
            
        # Stack the RNN layers
        for i in range(rnn_layers):
            current_layer += 1
            return_seq = (current_layer < total_layers)

            model.add(SimpleRNN(units=units, return_sequences=return_seq))
            model.add(Dropout(rate=dropout_rate))
        
    # Dense output layer with linear activation function
    model.add(Dense(units=1, activation='linear'))
    
    model = compile_model(model, learning_rate=0.001)
    
    return model

# Test building specific combinations for RNN + GRU
if __name__ == "__main__":
    test_combos = [
        (2, 1, 'rnn_first', "2 RNN + 1 GRU"),
        (3, 1, 'rnn_first', "3 RNN + 1 GRU"),
        (3, 2, 'rnn_first', "3 RNN + 2 GRU"),
        (2, 1, 'gru_first', "1 GRU + 2 RNN"),
        (2, 2, 'gru_first', "2 GRU + 2 RNN")
    ]
    
    for r_count, g_count, order, name in test_combos:
        print(f"\n--- Architecture: {name} ---")
        test_model = build_rnn_gru_model(rnn_layers=r_count, gru_layers=g_count, stack_order=order)
        test_model.summary()