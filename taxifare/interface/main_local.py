import numpy as np
import pandas as pd
from google.cloud import bigquery
from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse
from taxifare.params import *

def preprocess_and_train(min_date:str = '2009-01-01', max_date:str = '2015-01-01') -> None:
    """
    - Query the raw dataset from Le Wagon Big Query dataset
    - Cache query result as local CSV if not exist locally
    - Clean and preprocess data
    - Train a Keras model on it
    - Save the model
    - Compute & save a validation performance metric
    """

    print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess_and_train" + Style.RESET_ALL)

    from taxifare.ml_logic.data import clean_data
    from taxifare.ml_logic.preprocessor import preprocess_features
    from taxifare.ml_logic.registry import save_model, save_results
    from taxifare.ml_logic.model import (compile_model, initialize_model, train_model)

    min_date = parse(min_date).strftime('%Y-%m-%d') # e.g '2009-01-01'
    max_date = parse(max_date).strftime('%Y-%m-%d') # e.g '2009-01-01'

    query = f"""
        SELECT {",".join(COLUMN_NAMES_RAW)}
        FROM {GCP_PROJECT_WAGON}.{BQ_DATASET}.raw_{DATA_SIZE}
        WHERE pickup_datetime BETWEEN '{min_date}' AND '{max_date}'
        ORDER BY pickup_datetime
        """

    # Retrieve `query` data from Big Query, or from `data_query_cache_path` if file alredy exists!
    data_query_cache_path = Path(LOCAL_DATA_PATH).joinpath("raw", f"query_{min_date}_{max_date}_{DATA_SIZE}.csv")
    data_query_cached_exists = data_query_cache_path.is_file()

    if data_query_cached_exists:
        print("Load data from local CSV...")
        # YOUR CODE HERE

    else:
        print("Load data from Querying Big Query server...")
        # YOUR CODE HERE

        # Save it locally to accelerate for next queries!
        data.to_csv(data_query_cache_path, header=True, index=False)

    # Clean data using data.py
    # YOUR CODE HERE

    # Create (X_train, y_train, X_val, y_val) without data leaks
    # No need for test sets, we'll report val metrics only
    split_ratio = 0.02 # About one month of validation data
    # YOUR CODE HERE

    # Create (X_train_processed, X_val_processed) using `preprocessor.py`
    # Luckily, our preprocessor is stateless: We can "fit_transform" both X_train and X_val without data-leaks!
    # YOUR CODE HERE

    # Train model on training set, using `model.py`
    model = None
    learning_rate = 0.0005
    batch_size = 256
    patience = 2

    # YOUR CODE HERE

    # Compute the validation metric (min val mae of the holdout set)
    val_mae = np.min(history.history['val_mae'])

    # Save trained model
    params = dict(
        learning_rate=learning_rate,
        batch_size=batch_size,
        patience=patience
    )

    save_results(params=params, metrics=dict(mae=val_mae))
    save_model(model=model)

    print("✅ preprocess_and_train() done")


def pred(X_pred: pd.DataFrame = None) -> np.ndarray:

    print(Fore.MAGENTA + "\n ⭐️ Use case: pred" + Style.RESET_ALL)

    from taxifare.ml_logic.registry import load_model
    from taxifare.ml_logic.preprocessor import preprocess_features

    if X_pred is None:
       X_pred = pd.DataFrame(dict(
           pickup_datetime=[pd.Timestamp("2013-07-06 17:18:00", tz='UTC')],
           pickup_longitude=[-73.950655],
           pickup_latitude=[40.783282],
           dropoff_longitude=[-73.984365],
           dropoff_latitude=[40.769802],
           passenger_count=[1],
       ))

    model = load_model()
    X_processed = preprocess_features(X_pred)
    y_pred = model.predict(X_processed)

    print(f"✅ pred() done")
    return y_pred


if __name__ == '__main__':
    try:
        preprocess_and_train()
        # preprocess()
        # train()
        pred()
    except:
        import sys
        import traceback

        import ipdb
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
