import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, SimpleRNN, LSTM, Dropout, Dense
from utils import compile_model 

# Stacks RNN layers then LSTM layers
def build_rnn_lstm_model(rnn_layers: int, lstm_layers: int, stack_order: str = 'rnn_first', units: int = 256, dropout_rate: float = 0.1, input_shape=(100, 20)):
    model = Sequential()

    # Define input layer shape and total layers
    model.add(Input(shape=input_shape))
    total_layers = rnn_layers + lstm_layers
    current_layer = 0
    
    if stack_order == 'rnn_first':
        # Stack the RNN layers
        for i in range(rnn_layers):
            current_layer += 1
            return_seq = (current_layer < total_layers)

            model.add(SimpleRNN(units=units, return_sequences=return_seq))
            model.add(Dropout(rate=dropout_rate))

        # Stack the LSTM layers
        for i in range(lstm_layers):
            current_layer += 1
            return_seq = (current_layer < total_layers)
            
            model.add(LSTM(units=units, return_sequences=return_seq))
            model.add(Dropout(rate=dropout_rate))
            
    elif stack_order == 'lstm_first':
        # Stack the LSTM layers
        for i in range(lstm_layers):
            current_layer += 1
            return_seq = (current_layer < total_layers)
            
            model.add(LSTM(units=units, return_sequences=return_seq))
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

# Test building specific combinations for RNN + LSTM
if __name__ == "__main__":
    test_combos = [
        (2, 1, 'rnn_first', "2 RNN + 1 LSTM"),
        (3, 1, 'rnn_first', "3 RNN + 1 LSTM"),
        (3, 2, 'rnn_first', "3 RNN + 2 LSTM"),
        (2, 1, 'lstm_first', "1 LSTM + 2 RNN"),
        (2, 2, 'lstm_first', "2 LSTM + 2 RNN")
    ]
    
    for r_count, l_count, order, name in test_combos:
        print(f"\n--- Architecture: {name} ---")
        test_model = build_rnn_lstm_model(rnn_layers=r_count, lstm_layers=l_count, stack_order=order)
        test_model.summary()