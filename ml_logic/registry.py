import glob
import os
import pickle
import time

import tensorflow as tf
from tensorflow import keras
from google.cloud import storage

from params import *


def save_results(metrics: dict) -> None:
    """
    Persist params & metrics locally on the hard drive at
    "{LOCAL_REGISTRY_PATH}/params/{current_timestamp}.pickle"
    "{LOCAL_REGISTRY_PATH}/metrics/{current_timestamp}.pickle"
    - (unit 03 only) if MODEL_TARGET='mlflow', also persist them on MLflow
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save metrics locally
    if metrics is not None:
        metrics_dir = os.path.join(LOCAL_REGISTRY_PATH, "metrics")
        if not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir)
        metrics_path = os.path.join(metrics_dir, timestamp + ".pickle")
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)
    print("✅ Metrics saved locally")


def save_model(model: keras.Model = None) -> None:
    """
    Persist trained model locally on the hard drive at f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.h5"
    - if MODEL_TARGET='gcs', also persist it in your bucket on GCS at "models/{timestamp}.h5" --> unit 02 only
    - if MODEL_TARGET='mlflow', also persist it on MLflow instead of GCS (for unit 0703 only) --> unit 03 only
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    model_dir = os.path.join(LOCAL_REGISTRY_PATH)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, timestamp)
    tf.saved_model.save(model, model_path)
    print("✅ Model saved locally")

    if MODEL_TARGET == "gcs":
        model_filename = model_path.split("/")[-1]
        client = storage.Client()
        bucket = client.bucket(BUCKET_MODEL)

        for root, dirs, files in os.walk(model_path):
            for filename in files:

                local_path = os.path.join(root, filename)

                relative_path = os.path.relpath(local_path, start=model_path)
                gcs_path = f"{model_filename}/{relative_path}"

                blob = bucket.blob(gcs_path)
                blob.upload_from_filename(local_path)

        print("✅ Model saved to GCS")

    return None


def load_model(stage="Production") -> keras.Model:
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    - or from GCS (most recent one) if MODEL_TARGET=='gcs'  --> for unit 02 only
    - or from MLFLOW (by "stage") if MODEL_TARGET=='mlflow' --> for unit 03 only

    Return None (but do not Raise) if no model is found

    """

    if MODEL_TARGET == "local":

        # Get the latest model version name by the timestamp on disk
        local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
        local_model_paths = glob.glob(f"{local_model_directory}/*")

        if not local_model_paths:
            return None

        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

        latest_model = keras.models.load_model(most_recent_model_path_on_disk)

        return latest_model

    elif MODEL_TARGET == "gcs":
        # 🎁 We give you this piece of code as a gift. Please read it carefully! Add a breakpoint if needed!

        client = storage.Client()
        blobs = list(client.get_bucket(BUCKET_MODEL).list_blobs(prefix="model"))

        try:
            latest_blob = max(blobs, key=lambda x: x.updated)
            latest_model_path_to_save = os.path.join(
                LOCAL_REGISTRY_PATH, latest_blob.name
            )
            latest_blob.download_to_filename(latest_model_path_to_save)

            latest_model = keras.models.load_model(latest_model_path_to_save)

            print("✅ Latest model downloaded from cloud storage")

            return latest_model
        except:
            print(f"\n❌ No model found in GCS bucket {BUCKET_MODEL}")

            return None

    else:
        return None
