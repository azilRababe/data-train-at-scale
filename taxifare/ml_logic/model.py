
from colorama import Fore, Style

import time
print(Fore.BLUE + "\nLoading tensorflow..." + Style.RESET_ALL)
start = time.perf_counter()

from tensorflow import keras
from keras import Model, Sequential, layers, regularizers, optimizers
from keras.callbacks import EarlyStopping

end = time.perf_counter()
print(f"\n✅ tensorflow loaded ({round(end - start, 2)} secs)")

from typing import Tuple

import numpy as np


def initialize_model(input_shape: tuple) -> Model:
    """
    Initialize the Neural Network with random weights
    """
    # YOUR CODE HERE

    print("✅ model initialized")

    return model


def compile_model(model: Model, learning_rate=0.0005) -> Model:
    """
    Compile the Neural Network
    """
    # YOUR CODE HERE

    print("✅ model compiled")
    return model


def train_model(model: Model,
                X: np.ndarray,
                y: np.ndarray,
                batch_size=256,
                patience=2,
                validation_data=None, # overrides validation_split
                validation_split=0.3) -> Tuple[Model, dict]:
    """
    Fit model and return a the tuple (fitted_model, history)
    """

    # YOUR CODE HERE

    print(f"✅ model trained on {len(X)} rows with with min val MAE: {round(np.min(history.history['val_mae']), 2)}")

    return model, history

