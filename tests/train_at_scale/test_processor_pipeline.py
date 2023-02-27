import numpy as np

def test_preprocess_features(fixture_cleaned_1k, fixture_processed_1k):
    from taxifare.ml_logic.preprocessor import preprocess_features
    fixture_X_cleaned = fixture_cleaned_1k.drop(columns=['fare_amount'])
    fixture_X_processed = fixture_processed_1k.to_numpy()[:,:-1]

    X_processed = preprocess_features(fixture_X_cleaned)
    assert X_processed.shape == fixture_X_processed.shape
    assert np.allclose(X_processed, fixture_X_processed, atol=1e-3)
