import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

def compile_model(model, learning_rate=0.001):
    # compiles any deep learning model using the Adam optimizer, 
    # 0.001 learning rate, and MSE loss function as defined in the paper.

    # initialize Adam optimizer w the learning rate=0.001
    optimizer = Adam(learning_rate=learning_rate)

    # compile the model using Mean Squared Error (MSE) as the loss function
    # and Mean Absolute Error (MAE) as an evaluation metric
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    return model

def get_callbacks(patience_early_stop=15, patience_lr=5):

    # eeturns standard callbacks: EarlyStopping and ReduceLROnPlateau 
    # to aid model convergence and prevent overfitting.

    # halts training if model performance stops improving
    early_stopping = EarlyStopping(
        monitor='loss', # monitor training loss
        patience=patience_early_stop,   # number of epochs with no improvement before stopping
        restore_best_weights=True,  # keep the weights from the epoch with the best performance
        verbose=1
    )

    # automatically reduces the learning rate when a metric has stopped improving
    # for subtle precise adjustments
    reduce_lr = ReduceLROnPlateau(
        monitor='loss', # monitor training loss
        factor=0.5, # factor by which the learning rate will be reduced
        patience=patience_lr,   # numer of epochs with no imrpovement before reducing LR
        min_lr=1e-6,    # lower bound on the learning rate
        verbose=1
    )
    
    return [early_stopping, reduce_lr]