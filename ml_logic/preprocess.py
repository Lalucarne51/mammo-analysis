from tcia_utils import nbia
import numpy as np
import dicom2jpg
import os
from google.cloud import storage


BUCKET_NAME = os.environ.get("BUCKET_NAME")

# storage_filename = "models/random_forest_model.joblib"
# local_filename = "model.joblib"

def download_and_preprocess():

    local_data = "../data/raw"
    data = "$BUCKET_NAME/data_raw/"
    dir_data_processed = "$BUCKET_NAME/data_processed/"
    manifest = "../manifest/CBIS-DDSM-All-doiJNLP-zzWs5zfZ.tcia"

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    # converts manifest to list of UIDs
    uids = nbia.manifestToList(manifest)

    buckt_img = 0
    for pict_dcm in uids:
        for i in range(buckt_img, buckt_img+2):
            nbia.downloadSeries(manifest, input_type = "manifest", number=2, format = "df",path=local_data)
            dicom2jpg.dicom2jpg(data, target_root=dir_data_processed)
            buckt_img = buckt_img+2

def normalize_image(dataset):
    dataset = dataset / 255.
    X_test = X_test / 255.

    return X_train, X_test


def dim_expansion(X_train, X_test):
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    return X_train, X_test

def resize_image(X_train, X_test):
    X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
    X_test = (X_test - X_test.min()) / (X_test.max() - X_train.min())

    return X_train, X_test
