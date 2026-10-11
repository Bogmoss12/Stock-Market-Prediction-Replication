import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, LSTM, GRU, Dropout, Dense
from utils import compile_model 

# Stacks LSTM layers then GRU layers
def build_lstm_gru_model(lstm_layers: int, gru_layers: int, stack_order: str = 'lstm_first', units: int = 256, dropout_rate: float = 0.1, input_shape=(100, 20)):
    model = Sequential()

    # Define input layer shape and total layers
    model.add(Input(shape=input_shape))
    total_layers = lstm_layers + gru_layers
    current_layer = 0

    if stack_order == 'lstm_first':
        # Stack the LSTM layers
        for i in range(lstm_layers):
            current_layer += 1
            return_seq = (current_layer < total_layers)
            
            model.add(LSTM(units=units, return_sequences=return_seq))
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
            
        # Stack the LSTM Layers
        for i in range(lstm_layers):
            current_layer += 1
            return_seq = (current_layer < total_layers)

            model.add(LSTM(units=units, return_sequences=return_seq))
            model.add(Dropout(rate=dropout_rate))

    # Dense output layer with linear activation function
    model.add(Dense(units=1, activation='linear'))

    model = compile_model(model, learning_rate=0.001)
    
    return model

# Test building the different combinations
if __name__ == "__main__":
    test_combos = [
        (2, 1, 'lstm_first', "2 LSTM + 1 GRU"),
        (2, 2, 'lstm_first', "2 LSTM + 2 GRU"),
        (1, 2, 'gru_first', "2 GRU + 1 LSTM") 
    ]
    
    for l_count, g_count, order, name in test_combos:
        print(f"\n--- Architecture: {name} ---")
        test_model = build_lstm_gru_model(lstm_layers=l_count, gru_layers=g_count, stack_order=order)
        test_model.summary()