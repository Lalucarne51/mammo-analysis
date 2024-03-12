
import os

from google.cloud import storage
BUCKET_NAME = os.environ.get("BUCKET_NAME")

def import_blob():


    storage_filename = "models/xgboost_model.joblib"
    local_filename = "model.joblib"

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(storage_filename)
    blob.download_to_filename(local_filename)



def upload_blob():
    storage_filename = "models/random_forest_model.joblib"
    local_filename = "model.joblib"

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(storage_filename)
    blob.upload_from_filename(local_filename)
