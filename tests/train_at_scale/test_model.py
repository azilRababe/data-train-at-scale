import pytest
import numpy as np
import os

def test_model_can_fit(fixture_processed_1k):

    from taxifare.ml_logic.model import initialize_model,compile_model, train_model
    fixture_X_processed = fixture_processed_1k.to_numpy()[:,:-1]
    fixture_y = fixture_processed_1k.to_numpy()[:,-1]
    model = initialize_model(fixture_X_processed.shape[1:])
    model = compile_model(model, learning_rate=0.001)
    model, history = train_model(model=model,
                                 X=fixture_X_processed,
                                 y=fixture_y,
                                 validation_split=0.3)

    assert min(history.history['loss']) < 220, "Your model does not seem to fit correctly"
