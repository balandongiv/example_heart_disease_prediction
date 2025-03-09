import tensorflow as tf
from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.layers import Dense
from heart_disease_prediction.config import LEARNING_RATE, EPOCHS

def create_nn_model():
    model = Sequential([
        Dense(64, activation='gelu', kernel_regularizer=regularizers.l2(0.0001)),
        Dense(64, activation='gelu', kernel_regularizer=regularizers.l2(0.0001)),
        Dense(32, activation='gelu', kernel_regularizer=regularizers.l2(0.0001)),
        Dense(16, activation='gelu', kernel_regularizer=regularizers.l2(0.0001)),
        Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.0001))
    ])

    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE),
        metrics=['accuracy']
    )

    return model
