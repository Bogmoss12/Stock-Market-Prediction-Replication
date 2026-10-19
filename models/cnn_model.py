import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dropout, Dense
from utils import compile_model

def build_cnn_model(cnn_layers: int = 2, filters: int = 64, kernel_size: int = 3, dropout_rate: float = 0.1, input_shape=(100, 20)):
    """
    Constructs a 1D Convolutional Neural Network (CNN) model for stock trend extraction.
    Accepts cnn_layers to match the batch comparison runner script.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    
    # Dynamically stack Conv1D + MaxPooling blocks based on cnn_layers
    for i in range(cnn_layers):
        f_size = filters * (2 ** i)  # Doubles filters for deeper layers
        model.add(Conv1D(filters=f_size, kernel_size=kernel_size, activation='relu', padding='same'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(rate=dropout_rate))
        
    model.add(Flatten())
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(units=1, activation='linear'))
    
    return compile_model(model, learning_rate=0.001)

if __name__ == "__main__":
    print("\n--- CNN Model Architecture Test ---")
    model = build_cnn_model()
    model.summary()