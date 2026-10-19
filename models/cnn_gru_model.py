import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Conv1D, MaxPooling1D, LSTM, GRU, Dropout, Dense
from utils import compile_model

def build_cnn_sequence_model(cnn_layers: int = 2, seq_type: str = 'lstm', seq_layers: int = 1, 
                             filters: int = 64, units: int = 128, dropout_rate: int = 0.1, input_shape=(100, 20)):
    model = Sequential()
    model.add(Input(shape=input_shape))
    
    # 1. Convolutional feature extraction blocks
    for i in range(cnn_layers):
        f_size = filters * (2 ** i)
        model.add(Conv1D(filters=f_size, kernel_size=3, activation='relu', padding='same'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(rate=dropout_rate))
        
    # 2. Sequential layers (LSTM or GRU)
    for i in range(seq_layers):
        # All layers except the last sequence layer must return sequences
        return_seq = (i < seq_layers - 1)
        
        if seq_type.lower() == 'lstm':
            model.add(LSTM(units=units, return_sequences=return_seq))
        elif seq_type.lower() == 'gru':
            model.add(GRU(units=units, return_sequences=return_seq))
            
        model.add(Dropout(rate=dropout_rate))
        
    model.add(Dense(units=1, activation='linear'))
    return compile_model(model, learning_rate=0.001)