import numpy as np

def test_clean_data(fixture_query_1k, fixture_cleaned_1k):
    from taxifare.ml_logic.data import clean_data
    df_cleaned = clean_data(fixture_query_1k)
    assert df_cleaned.shape == fixture_cleaned_1k.shape
