### RESUME

LOCAL_DATA_PATH = Path('~').joinpath(".lewagon", "mlops", "data").expanduser()
GCP_PROJECT_WAGON = "wagon-public-datasets"
BQ_DATASET = "taxifare"
DATA_SIZE = "200k"  # raw_200k is a randomly sampled materialized view from "raw_all" data table
MIN_DATE = '2009-01-01'
MAX_DATE = '2015-01-01'
COLUMN_NAMES_RAW = ('fare_amount',	'pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count')

GCP_PROJECT = "<your gcp project id>"


query = f"""
    SELECT {",".join(COLUMN_NAMES_RAW)}
    FROM {GCP_PROJECT_WAGON}.{BQ_DATASET}.raw_{DATA_SIZE}
    WHERE pickup_datetime BETWEEN '{MIN_DATE}' AND '{MAX_DATE}'
    ORDER BY pickup_datetime
    """
print(query)




##########################################

# CODE CHUNK BY CHUNK

def preprocess(min_date: str = '2009-01-01', max_date: str = '2015-01-01') -> None:
    """
    1. Query and preprocess the raw dataset iteratively (in chunks)
    2. Store the newly processed (and raw) data on your local hard drive for later use

    - If raw data already exists on your local disk, use `pd.read_csv(..., chunksize=CHUNK_SIZE)`
    - If raw data does NOT yet exist, use `bigquery.Client().query().result().to_dataframe_iterable()`
    """
    print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess by batch" + Style.RESET_ALL)

    min_date = parse(min_date).strftime('%Y-%m-%d') # e.g '2009-01-01'
    max_date = parse(max_date).strftime('%Y-%m-%d') # e.g '2009-01-01'

    query = f"""
        SELECT {",".join(COLUMN_NAMES_RAW)}
        FROM {GCP_PROJECT_WAGON}.{BQ_DATASET}.raw_{DATA_SIZE}
        WHERE pickup_datetime BETWEEN '{min_date}' AND '{max_date}'
        ORDER BY pickup_datetime
        """
    # Retrieve `query` data as a DataFrame iterable
    data_query_cache_path = Path(LOCAL_DATA_PATH).joinpath("raw", f"query_{min_date}_{max_date}_{DATA_SIZE}.csv")
    data_processed_path = Path(LOCAL_DATA_PATH).joinpath("processed", f"processed_{min_date}_{max_date}_{DATA_SIZE}.csv")

    data_query_cache_exists = data_query_cache_path.is_file()
    if data_query_cache_exists:
        print("Get a DataFrame iterable from local CSV...")
        chunks = None

        # $CODE_BEGIN
        chunks = pd.read_csv(
            data_query_cache_path,
            chunksize=CHUNK_SIZE,
            parse_dates=["pickup_datetime"]
        )
        # $CODE_END
    else:
        print("Get a DataFrame iterable from querying the BigQuery server...")
        chunks = None

        # 🎯 HINT: `bigquery.Client(...).query(...).result(page_size=...).to_dataframe_iterable()`
        # $CODE_BEGIN
        client = bigquery.Client(project=GCP_PROJECT)

        query_job = client.query(query)
        result = query_job.result(page_size=CHUNK_SIZE)

        chunks = result.to_dataframe_iterable()
        # $CODE_END

    for chunk_id, chunk in enumerate(chunks):
        print(f"Processing chunk {chunk_id}...")

        # Clean chunk
        # $CODE_BEGIN
        chunk_clean = clean_data(chunk)
        # $CODE_END

        # Create chunk_processed
        # 🎯 HINT: create (`X_chunk`, `y_chunk`), process only `X_processed_chunk`, then concatenate (X_processed_chunk, y_chunk)
        # $CODE_BEGIN
        X_chunk = chunk_clean.drop("fare_amount", axis=1)
        y_chunk = chunk_clean[["fare_amount"]]
        X_processed_chunk = preprocess_features(X_chunk)

        chunk_processed = pd.DataFrame(np.concatenate((X_processed_chunk, y_chunk), axis=1))
        # $CODE_END

        # Save and append the processed chunk to a local CSV at "data_processed_path"
        # 🎯 HINT: df.to_csv(mode=...)
        # 🎯 HINT: we want a CSV with neither index nor headers (they'd be meaningless)
        # $CODE_BEGIN
        chunk_processed.to_csv(
            data_processed_path,
            mode="w" if chunk_id==0 else "a",
            header=False,
            index=False,
        )
        # $CODE_END

        # Save and append the raw chunk `if not data_query_cache_exists`
        # $CODE_BEGIN
        # 🎯 HINT: we want a CSV with headers this time
        # 🎯 HINT: only the first chunk should store headers
        if not data_query_cache_exists:
            chunk.to_csv(
                data_query_cache_path,
                mode="w" if chunk_id==0 else "a",
                header=True if chunk_id==0 else False,
                index=False
            )
        # $CODE_END

    print(f"✅ data query saved as {data_query_cache_path}")
    print("✅ preprocess() done")


def train(min_date:str = '2009-01-01', max_date:str = '2015-01-01') -> None:
    """
    Incremental training on the (already preprocessed) dataset, stored locally

    - Loading data in chunks
    - Updating the weight of the model for each chunk
    - Saving validation metrics at each chunk, and final model weights on the local disk
    """

    print(Fore.MAGENTA + "\n ⭐️ Use case: train in batches" + Style.RESET_ALL)

    data_processed_path = Path(LOCAL_DATA_PATH).joinpath("processed", f"processed_{min_date}_{max_date}_{DATA_SIZE}.csv")
    model = None
    metrics_val_list = []  # store the val_mae of each chunk

    # Iterate in chunks and partially fit on each chunk
    chunks = pd.read_csv(
        data_processed_path,
        chunksize=CHUNK_SIZE,
        header=None,
        dtype=DTYPES_PROCESSED
    )

    for chunk_id, chunk in enumerate(chunks):
        print(f"Training on preprocessed chunk n°{chunk_id}")

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

        # Train a model *incrementally*, and store the val_mae of each chunk in `metrics_val_list`
        # $CODE_BEGIN
        if model is None:
            model = initialize_model(input_shape=X_train_chunk.shape[1:])

        model = compile_model(model, learning_rate)

        model, history = train_model(
            model,
            X_train_chunk,
            y_train_chunk,
            batch_size=batch_size,
            patience=patience,
            validation_data=(X_val_chunk, y_val_chunk)
        )

        metrics_val_chunk = np.min(history.history['val_mae'])
        metrics_val_list.append(metrics_val_chunk)

        print(metrics_val_chunk)
        # $CODE_END

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
# $ERASE_END

##########################################
