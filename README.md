# ⛰ "Train At Scale" Unit 🗻

In this unit, you will learn how to package the notebook provided by the Data Science team at WagonCab, and how to scale it so that it can be trained locally on the full dataset.

This unit consists of the 5 challenges below, they are all grouped up in this single `README` file.

Simply follow the guide and `git push` after each main section so we can track your progress!

# 1️⃣ Local Setup

<details>
  <summary markdown='span'><strong>❓Instructions (expand me)</strong></summary>

As lead ML Engineer for the project, your first role is to set up a local working environment (with `pyenv`) and a python package that only contains the skeleton of your code base.

💡 Packaging notebooks is a key ML Engineer skill. It allows
- other users to collaborate on the code
- you to clone the code locally or on a remote machine to, for example, train the `taxifare` model on a more powerful machine
- you to put the code in production (on a server that never stops running) to expose it as an **API** or through a **website**
- you to render the code operable so that it can be run manually or plugged into an automation workflow

### 1.1) Create a new pyenv called [🐍 taxifare-env]

🐍 Create the virtual env

```bash
cd ~/code/<user.github_nickname>/{{local_path_to("07-ML-Ops/01-Train-at-scale/01-Train-at-scale")}}
python --version # First, check your Python version for <YOUR_PYTHON_VERSION> below (e.g. 3.10.6)
```

```bash
pyenv virtualenv <YOUR_PYTHON_VERSION> taxifare-env
pyenv local taxifare-env
pip install --upgrade pip
code .
```

Then, make sure both your OS' Terminal and your VS Code's integrated Terminal display `[🐍 taxifare-env]`.
In VS code, open any `.py` file and check that `taxifare-env` is activated by clicking on the pyenv section in the bottom right, as seen below:

<a href="https://wagon-public-datasets.s3.amazonaws.com/data-science-images/07-ML-OPS/pyenv-setup.png" target="_blank">
    <img src='https://wagon-public-datasets.s3.amazonaws.com/data-science-images/07-ML-OPS/pyenv-setup.png' width=400>
</a>

### 1.2) Get familiar with the taxifare package structure

❗️Take 10 minutes to understand the structure of the boilerplate we've prepared for you (don't go into detail); its entry point is `taxifare.interface.main_local`: follow it quickly.

```bash
. # Challenge folder root
├── Makefile          # Main "interface" with your project; use it to launch tests, start trainings, etc. from the CLI
├── README.md         # The file you are reading right now!
├── notebooks
│   └── datascientist_deliverable.ipynb   # The deliverable from the DS team!
├── pytest.ini         # Test configuration file (please do not touch)
├── requirements.txt   # List all third-party packages to add to your local environment
├── setup.py           # Enable `pip install` for your package
├── taxifare           # The code logic for this package
│   ├── __init__.py
│   ├── interface
│   │   ├── __init__.py
│   │   └── main_local.py  # Your main Python entry point containing all "routes"
│   └── ml_logic
│   |    ├── __init__.py
│   |    ├── data.py           # Save, load and clean data
│   |    ├── encoders.py       # Custom encoder utilities
│   |    ├── model.py          # TensorFlow model
│   |    ├── preprocessor.py   # Sklearn preprocessing pipelines
│   |    ├── registry.py       # Save and load models
|   ├── utils.py    # # Useful python functions with no dependencies on taxifare logic
|   ├── params.py   # Global project params
|
├── tests  # Tests to run using `make pytest`
│   ├── ...
│   └── ...
├── .gitignore
```

🐍 Install your package on this new virtual env

```bash
cd ~/code/<user.github_nickname>/{{local_path_to("07-ML-Ops/01-Train-at-scale/01-Train-at-scale")}}
pip install -e .
```

Make sure the package is installed by running `pip list | grep taxifare`; it should print the absolute path to the package.


### 1.3) Where is the data?

**Raw data is in Google Big Query**

WagonCab's engineering team stores all it's cab course history since 2009 in a massive Big Query table `wagon-public-datasets.taxifare.raw_all`.
- This table contains `1.1 Million` for this challenge exactly, from **2009 to jun 2015**.
- *(Note from Le Wagon: In reality, there is 55M rows but we limited that for cost-control in the whole module)*

**Check access to Google Cloud Platform**
Your computer should already be configured to have access to Google Cloud Platform since [setup-day](https://github.com/lewagon/data-setup/blob/master/macOS.md#google-cloud-platform-setup)

🧪 Check that everything is fine
```bash
make test_gcp_setup
```

**We'll always cache all intermediate data locally in `~/.lewagon/mlops/` to avoid querying BQ twice**

💾 Let's store our `data` folder *outside* of this challenge folder so that it can be accessed by all other challenges throughout the whole ML Ops module. We don't want it to be tracked by `git` anyway!

``` bash
# Create the data folder
mkdir -p ~/.lewagon/mlops/data/

# Create relevant subfolders
mkdir ~/.lewagon/mlops/data/raw
mkdir ~/.lewagon/mlops/data/processed
```

💡While we are here, let's also create a storage folder for our `training_outputs` that will also be shared by all challenges

```bash
# Create the training_outputs folder
mkdir ~/.lewagon/mlops/training_outputs

# Create relevant subfolders
mkdir ~/.lewagon/mlops/training_outputs/metrics
mkdir ~/.lewagon/mlops/training_outputs/models
mkdir ~/.lewagon/mlops/training_outputs/params
```

You can now see that the data for the challenges to come is stored in `~/.lewagon/mlops/`, along with the notebooks of the Data Science team and the model outputs:

``` bash
tree -a ~/.lewagon/mlops/

# YOU SHOULD SEE THIS
├── data          # This is where you will:
│   ├── processed # Store intermediate, processed data
│   └── raw       # Download samples of the raw data
└── training_outputs
    ├── metrics # Store trained model metrics
    ├── models  # Store trained model weights (can be large!)
    └── params  # Store trained model hyperparameters
```

☝️ Feel free to remove all files but keep this empty folder structure at any time using

```bash
make reset_local_files
```

</details>

# 2️⃣ Understand the Work of a Data Scientist

<details>
  <summary markdown='span'><strong>❓Instructions (expand me)</strong></summary>

*⏱ Duration:  spend 1 hour on this*

🖥️ Open `datascientist_deliverable.ipynb` with VS Code (forget about Jupyter for this module), and run all cells carefully, while understanding them. This handover between you and the DS team is the perfect time to interact with them (i.e. your buddy or a TA).

❗️Make sure to use `taxifare-env` as an `ipykernel` venv

<a href="https://wagon-public-datasets.s3.amazonaws.com/data-science-images/07-ML-OPS/pyenv-notebook.png" target="_blank">
    <img src='https://wagon-public-datasets.s3.amazonaws.com/data-science-images/07-ML-OPS/pyenv-notebook.png' width=400>
</a>

</details>


# 3️⃣ Package Your Code

<details>
  <summary markdown='span'><strong>❓Instructions (expand me)</strong></summary>

🎯 Your goal is to be able to run the `taxifare.interface.main_local` module as seen below

```bash
# -> model
python -m taxifare.interface.main_local
```

🖥️ To do so, please code the missing parts marked with `# YOUR CODE HERE` in the following files; it should follow the Notebook pretty closely!

```bash
├── taxifare
│   ├── __init__.py
│   ├── interface
│   │   ├── __init__.py
│   │   └── main_local.py   # 🔵 🚪 Entry point: code both `preprocess_and_train()` and `pred()`
│   └── ml_logic
│       ├── __init__.py
│       ├── data.py          # 🔵 your code here
│       ├── encoders.py      # 🔵 your code here
│       ├── model.py         # 🔵 your code here
│       ├── preprocessor.py  # 🔵 your code here
│       ├── registry.py  # ✅ `save_model` and `load_model` are already coded for you
|   ├── params.py # 🔵 You need to fill your GCP_PROJECT
│   ├── utils.py
```

**🧪 Test your code**

Make sure you have the package installed correctly in your current taxifare-env, if not

```bash
pip list | grep taxifare
```

Then, make sure your package runs properly with `python -m taxifare.interface.main_local`.
- Debug it until it runs!
- Use the following dataset sizes

```python
# taxifare/ml_logic/params.py
DATA_SIZE = '1k'   # To iterate faster in debug mode 🐞
DATA_SIZE = '200k' # Should work at least once
# DATA_SIZE = 'all' 🚨 DON'T TRY YET, it's too big and will cost money!
```

Then, only try to pass tests with `make test_preprocess_and_train`!

✅ When you are all green, track your results on kitt with `make test_kitt`

</details>

# 4️⃣ Investigate Scalability

<details>
  <summary markdown='span'><strong>❓Instructions (expand me)</strong></summary>

*⏱ Duration:  spend 20 minutes at most on this*

Now that you've managed to make the package work for a small dataset, time to see how it will handle the real dataset!

👉 Change `ml_logic.params.DATA_SIZE` to `all` to start getting serious!

🕵️ Investigate which part of your code takes **the most time** and uses **the most memory**  using `ml_logic.utils.simple_time_and_memory_tracker` to decorate the methods of your choice.

```python
# taxifare.ml_logic.data.py
from taxifare.ml_logic.utils import simple_time_and_memory_tracker

@simple_time_and_memory_tracker
def clean_data() -> pd.DataFrame:
    ...
```

💡 If you don't remember exactly how decorators work, refer to our [04/05-Communicate](https://kitt.lewagon.com/camps/<user.batch_slug>/lectures/content/04-Decision-Science_05-Communicate.slides.html?title=Communicate#/6/3) lecture!

🕵️ Try to answer the following questions with your buddy:
- What part of your code holds the key bottlenecks ?
- What kinds of bottlenecks are the most worrying? (time? memory?)
- Do you think it will scale if we had given you the 50M rows ? 500M ? By the way, the [real NYC dataset](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page) is even bigger and weights in at about 156GB!
- Can you think about potential solutions? Write down your ideas, but do not implement them yet!
</details>


# 5️⃣ Incremental Processing

<details>
  <summary markdown='span'><strong>❓Instructions (expand me)</strong></summary>

🎯 Your goal is to improve your codebase to be able to train the model on unlimited amount of rows, **without reaching RAM limits**, on a single computer.

## 5.1) Discussion

**What did we learn?**

We have memory and time constraints:
- A `(55M, 8)`-shaped raw data gets loaded into memory as a DataFrame and takes up about 10GB of RAM, which is too much for most computers.
- A `(55M, 65)`-shaped preprocessed DataFrame is even bigger.
- The `ml_logic.encoders.compute_geohash` method takes a very long time to process 🤯

One solution is to pay for a *cloud Virtual Machine (VM)* with enough RAM and process it there (this is often the simplest way to deal with such a problem).

**Proposed solution: incremental preprocessing 🔪 chunk by chunk 🔪**

<img src="https://wagon-public-datasets.s3.amazonaws.com/data-science-images/07-ML-OPS/process_by_chunk.png" width=500>

💡 As our preprocessor is *stateless*, we can easily:
- Avoid computing any _column-wise statistics_ but only perform _row-by-row preprocessing_
- Decouple the _preprocessing_ from the _training_ and store any intermediate results on disk!

🙏 Therefore, let's do the preprocessing *chunk by chunk*, with chunks of limited size (e.g. 100.000 rows), each chunk fitting nicely in memory:

1. We'll store `data_processed_chunk_01` on a hard-drive.
2. Then append `data_processed_chunk_02` to the first.
3. etc...
4. Until a massive CSV is stored at `~/.lewagon/mlops/data/processed/processed_all.csv`

5. In section 6️⃣, we'll `train()` our model chunk-by-chunk too by loading & training iteratively on each chunk (more on that next section)

## 5.2) Your turn: code `def preprocess()`

👶 **First, let's bring back smaller dataset sizes for debugging purposes**

```python
# params.py
DATA_SIZE = '1k'
CHUNK_SIZE = 200
```

**Then, code the new route given below by `def preprocess()` in your `ml_logic.interface.main_local` module; copy and paste the code below to get started**

[//]: # (  🚨 Code below is NOT the single source of truth. Original is in data-solutions repo 🚨 )

<br>

<details>
  <summary markdown='span'>👇 Code to copy 👇</summary>

```python
def preprocess(min_date: str = '2009-01-01', max_date: str = '2015-01-01') -> None:
    """
    Query and preprocess the raw dataset iteratively (by chunks).
    Then store the newly processed (and raw) data on local hard-drive for later re-use.

    - If raw data already exists on local disk:
        - use `pd.read_csv(..., chunksize=CHUNK_SIZE)`

    - If raw data does not yet exists:
        - use `bigquery.Client().query().result().to_dataframe_iterable()`

    """
    print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess by batch" + Style.RESET_ALL)

    from taxifare.ml_logic.data import clean_data
    from taxifare.ml_logic.preprocessor import preprocess_features

    min_date = parse(min_date).strftime('%Y-%m-%d') # e.g '2009-01-01'
    max_date = parse(max_date).strftime('%Y-%m-%d') # e.g '2009-01-01'

    query = f"""
        SELECT {",".join(COLUMN_NAMES_RAW)}
        FROM {GCP_PROJECT_WAGON}.{BQ_DATASET}.raw_{DATA_SIZE}
        WHERE pickup_datetime BETWEEN '{min_date}' AND '{max_date}'
        ORDER BY pickup_datetime
        """
    # Retrieve `query` data as dataframe iterable
    data_query_cache_path = Path(LOCAL_DATA_PATH).joinpath("raw", f"query_{min_date}_{max_date}_{DATA_SIZE}.csv")
    data_processed_path = Path(LOCAL_DATA_PATH).joinpath("processed", f"processed_{min_date}_{max_date}_{DATA_SIZE}.csv")

    data_query_cache_exists = data_query_cache_path.is_file()
    if data_query_cache_exists:
        print("Get a dataframe iterable from local CSV...")
        chunks = None
        # YOUR CODE HERE

    else:
        print("Get a dataframe iterable from Querying Big Query server...")
        chunks = None
        # 🎯 Hints: `bigquery.Client(...).query(...).result(page_size=...).to_dataframe_iterable()`
        # YOUR CODE HERE

    for chunk_id, chunk in enumerate(chunks):
        print(f"processing chunk {chunk_id}...")

        # Clean chunk
        # YOUR CODE HERE

        # Create chunk_processed
        # 🎯 Hints: Create (`X_chunk`, `y_chunk`), process only `X_processed_chunk`, then concatenate (X_processed_chunk, y_chunk)
        # YOUR CODE HERE

        # Save and append the processed chunk to a local CSV at "data_processed_path"
        # 🎯 Hints: df.to_csv(mode=...)
        # 🎯 Hints: We want a CSV without index nor headers (they'd be meaningless)
        # YOUR CODE HERE

        # Save and append the raw chunk if not `data_query_cache_exists`
        # YOUR CODE HERE
    print(f"✅ data query saved as {data_query_cache_path}")
    print("✅ preprocess() done")


```

</details>

<br>

**❓Try to create and store the following preprocessed datasets**

- `data/processed/train_processed_1k.csv` by running `preprocess()`

<br>

**🧪 Test your code**

Test your code with `make test_preprocess_by_chunk`.

✅ When you are all green, track your results on kitt with `make test_kitt`

<br>

**❓Finally, create and store the real preprocessed datasets**

Using:
```python
# params.py
DATA_SIZE = 'all'
CHUNK_SIZE = 100000
```

🎉 Given a few hours of computation, we could easily process the 55 Million rows too, but let's not do it today 😅

</details>

# 6️⃣ Incremental Learning

<details>
  <summary markdown='span'><strong>❓Instructions (expand me)</strong></summary>

<br>

🎯 Goal: train our model on the full `.../processed/processed_all.csv`

## 6.1) Discussion

In theory, we cannot load such a big dataset of shape `(xxMillions, 65)` into RAM all at once, but we can load it in chunks.

**How do we train a model in chunks?**

This is called **incremental learning**, or **partial_fit**
- We initialize a model with random weights ${\theta_0}$
- We load the first `data_processed_chunk` into memory (say, 100_000 rows)
- We train our model on the first chunk and update its weights accordingly ${\theta_0} \rightarrow {\theta_1}$
- We load the second `data_processed_chunk` into memory
- We *retrain* our model on the second chunk, this time updating the previously computed weights ${\theta_1} \rightarrow {\theta_2}$!
- We rinse and repeat until the end of the dataset

❗️Not all Machine Learning models support incremental learning; only *parametric* models $f_{\theta}$ that are based on *iterative update methods* like Gradient Descent support it
- In **scikit-learn**, `model.partial_fit()` is only available for the SGDRegressor/Classifier and a few others ([read this carefully 📚](https://scikit-learn.org/0.15/modules/scaling_strategies.html#incremental-learning)).
- In **TensorFlow** and other Deep Learning frameworks, training is always iterative, and incremental learning is the default behavior! You just need to avoid calling `model.initialize()` between two chunks!

❗️Do not confuse `chunk_size` with `batch_size` from Deep Learning

👉 For each (big) chunk, your model will read data in many (small) batches over several epochs

<img src='https://wagon-public-datasets.s3.amazonaws.com/data-science-images/07-ML-OPS/train_by_chunk.png'>

👍 **Pros:** this universal approach is framework-independent; you can use it with `scikit-learn`, XGBoost, TensorFlow, etc.

👎 **Cons:** the model will be biased towards fitting the *latest* chunk better than the *first* ones. In our case, it is not a problem as our training dataset is shuffled, but it is important to keep that in mind when we do a partial fit of our model with newer data once it is in production.

<br>

<details>
  <summary markdown='span'><strong>🤔 Do we really need chunks with TensorFlow?</strong></summary>

Granted, thanks to TensorFlow datasets you will not always need "chunks" as you can use batch-by-batch dataset loading as seen below

```python
import tensorflow as tf

ds = tf.data.experimental.make_csv_dataset(data_processed_all.csv, batch_size=256)
model.fit(ds)
```

We will see that in Recap. Still, in this challenge, we would like to teach you the universal method of incrementally fitting in chunks, as it applies to any framework, and will prove useful to *partially retrain* your model with newer data once it is put in production.
</details>

<br>

## 6.2) Your turn - code `def train()`

**Try to code the new route given below by `def train()` in your `ml_logic.interface.main_local` module; copy and paste the code below to get started**

Again, start with a very small dataset size, then finally train your model on 500k rows.

[//]: # (  🚨 Code below is not the single source of truth 🚨 )

<details>
  <summary markdown='span'><strong>👇 Code to copy 👇</strong></summary>

```python
def train(min_date:str = '2009-01-01', max_date:str = '2015-01-01') -> None:
    """
    Incremental train on the (already preprocessed) dataset locally stored.
    - Loading data chunk-by-chunk
    - Updating the weight of the model for each chunk
    - Saving validation metrics at each chunks, and final model weights on local disk
    """

    print(Fore.MAGENTA + "\n ⭐️ Use case:train by batch" + Style.RESET_ALL)
    from taxifare.ml_logic.registry import save_model, save_results
    from taxifare.ml_logic.model import (compile_model, initialize_model, train_model)

    data_processed_path = Path(LOCAL_DATA_PATH).joinpath("processed", f"processed_{min_date}_{max_date}_{DATA_SIZE}.csv")
    model = None
    metrics_val_list = []  # store each val_mae of each chunk

    # Iterate in chunks and partial fit on each chunk
    chunks = pd.read_csv(data_processed_path,
                         chunksize=CHUNK_SIZE,
                         header=None,
                         dtype=DTYPES_PROCESSED)

    for chunk_id, chunk in enumerate(chunks):
        print(f"training on preprocessed chunk n°{chunk_id}")
        # You can adjust training params for each chunk if you want!
        learning_rate = 0.0005
        batch_size = 256
        patience=2
        split_ratio = 0.1 # Higher train/val split ratio when chunks are small! Feel free to adjust.

        # Create (X_train_chunk, y_train_chunk, X_val_chunk, y_val_chunk)
        train_length = int(len(chunk)*(1-split_ratio))
        chunk_train = chunk.iloc[:train_length, :].sample(frac=1).to_numpy()
        chunk_val = chunk.iloc[train_length:, :].sample(frac=1).to_numpy()

        X_train_chunk = chunk_train[:, :-1]
        y_train_chunk = chunk_train[:, -1]
        X_val_chunk = chunk_val[:, :-1]
        y_val_chunk = chunk_val[:, -1]

        # Train a model *incrementally*, and store the val MAE of each chunk in `metrics_val_list`
        # YOUR CODE HERE

    # Return the last value of the validation MAE
    val_mae = metrics_val_list[-1]

    # Save model and training params
    params = dict(
        learning_rate=learning_rate,
        batch_size=batch_size,
        patience=patience,
        incremental=True,
        chunk_size=CHUNK_SIZE
    )

    print(f"✅ Trained with MAE: {round(val_mae, 2)}")

     # Save results & model
    save_results(params=params, metrics=dict(mae=val_mae))
    save_model(model=model)

    print("✅ train() done")

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

```

</details>

**🧪 Test your code**

Check it out with `make test_train_by_chunk`

✅ When you are all green, track your results on kitt with `make test_kitt`

🏁 🏁 🏁 🏁 Congratulations! 🏁 🏁 🏁 🏁


</details>
