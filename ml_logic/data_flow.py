# Construction du DataFrame ['image_id', 'array', 'cancer'] depuis les données de base

import pandas as pd
import numpy as np

import os
from google.cloud import storage
import cv2
from PIL import Image
import tensorflow as tf
import tensorflow_io as tfio
from tensorflow.data import Dataset


CUR_DIR = os.getcwd()
PROJECT_DIR = os.path.dirname(CUR_DIR)
DATA_DIR = os.path.join(PROJECT_DIR,'data')
METADATA_DIR = os.path.join(DATA_DIR, 'metadata')
IMAGE_DIR = os.path.join(DATA_DIR,'images')
PROCESSED_DIR = os.path.join(IMAGE_DIR,'data_processed')
DATA_CLEAN = os.path.join(IMAGE_DIR, 'clean')
file = 'kaggle.xlsx'

DIM = os.environ.get('DIM')

BUCKET_NAME = os.environ.get('BUCKET_NAME')
BUCKET_256 = os.environ.get('BUCKET_256')
GCP_PROJECT = os.environ.get('GCP_PROJECT')

# Initialize the GCS client
client = storage.Client(project=GCP_PROJECT)
bucket = client.bucket(BUCKET_NAME)

# Create a folder 'clean_data' if it doesn't exist in the bucket 'mammo_data'
if 'clean_data/' not in list(client.list_buckets()):
    blob = bucket.blob('clean_data/')
    blob.upload_from_string('')
    blobs = list(client.list_buckets())
    index = blobs.index('clean_data/')
    CLEAN_DATA = blobs[index]

def preproc_image(path:str, dim=DIM, nb_img=None, extension='.jpg'):
    '''
    Load, Resize, Convert into np.Array all images (.jpg) from a path and Save new images into a "path/clean" folder
    '''

    blobs = list(bucket.list_blobs())
    for blob in blobs:
        if blob.endswith(extension):
            img = cv2.imread(blob)
            resize_img = tf.image.resize(img, DIM, method='nearest')
            blob_name = f"{os.path.splitext(blob)[0]}{extension}"
            cv2.imwrite(os.path.join(CLEAN_DATA, blob_name), resize_img)


def excel_to_df(path):
    '''
    Read the file 'kaggle.xlsx' to make a DataFrame with 'image_id', 'array'
    '''

    image_id_array = []
    image_id_jepg = []
    image_array_path = []
    image_jpeg_path = []

    metadata = pd.read_excel(os.path.join(METADATA_DIR, file))
    metadata = metadata[['cancer', 'image_id']]

    for img in list(bucket.list_blobs()):
        image_id_array.append(np.int64(int(os.path.splitext(img)[0])))
        image_array_path.append(os.path.join(CLEAN_DATA, img))

    image_array_csv = pd.DataFrame(dict(
        image_id=image_id_array,
        image_array_path=image_array_path
    ))

    for img in list(bucket.list_blobs()):
        if img.endswith('.jpg'):
            image_id_jepg.append(np.int64(int(os.path.splitext(img)[0])))
            image_jpeg_path.append(os.path.join(BUCKET_NAME, img))

    image_jpeg_csv = pd.DataFrame(dict(
    image_id=image_id_jepg,
    image_jpeg_path=image_jpeg_path
    ))

    final_csv = image_array_csv.merge(metadata, on= 'image_id', how='left')
    final_csv = image_jpeg_csv.merge(final_csv, on= 'image_id', how='left')
    final_csv.to_csv(os.path.join(METADATA_DIR, 'final.csv'))

def create_dataset(path):
    '''
    Create a Dataset from the 'final.csv'
    '''

    variables = pd.read_csv(os.path.join(METADATA_DIR, 'final.csv'), index_col= None, header=[0])

    labels = []
    images_ids = []
    actual_images = []
    resize_images = []

    for idx, image_id, image_jpeg_path, image_array_path, label in variables.values:
        # Get labels
        labels.append(label)

        # Get images
        img = tf.io.read_file(image_jpeg_path)
        img = tf.io.decode_jpeg(img)

        actual_images.append(img)

        # Resize images
        img = tf.image.resize(img, [DIM, DIM], method='nearest')
        resize_images.append(img)

        # Labels to Tensor
        labels_tensor = tf.cast(labels, dtype= tf.int32)
        labels_tensor = tf.constant(labels_tensor, shape=(1667, 1, 1))

        # Images to Tensor & Normalize images
        image_tensor = tf.cast(resize_images, dtype= tf.float32) / 255
        image_tensor = tf.constant(image_tensor, shape=(1667, 1, DIM, DIM, 1))

        dataset = tf.data.Dataset.from_tensor_slices((image_tensor, labels_tensor))

        return dataset

def train_val_test_split(dataset, train_ratio=0.8, test_ratio=0.95):
    '''
    Create a training, a validation and a test dataset from the previous dataset.
    '''

    cancerous_dataset = dataset.map(lambda x: x[x['cancer']==1])
    normalous_dataset = dataset.map(lambda x: x[x['cancer']==0])

    train_cancerous_dataset = cancerous_dataset.take(int(len(cancerous_dataset)*train_ratio))
    test_cancerous_dataset = cancerous_dataset.skip(int(len(cancerous_dataset)*test_ratio))
    concat_cancerous_dataset = train_cancerous_dataset.concatenate(test_cancerous_dataset)
    iterable_cancerous_dataset = set(concat_cancerous_dataset.as_numpy_iterator())
    val_cancerous_dataset = cancerous_dataset.filter(lambda x: x.numpy() not in iterable_cancerous_dataset)
    train_val_cancerous_dataset = train_cancerous_dataset.concatenate(val_cancerous_dataset)

    train_normalous_dataset = normalous_dataset.take(int(len(normalous_dataset)*train_ratio))
    test_normalous_dataset = normalous_dataset.skip(int(len(normalous_dataset)*test_ratio))
    concat_normalous_dataset = train_normalous_dataset.concatenate(test_normalous_dataset)
    iterable_normalous_dataset = set(concat_normalous_dataset.as_numpy_iterator())
    val_normalous_dataset = normalous_dataset.filter(lambda x: x.numpy() not in iterable_normalous_dataset)
    train_val_normalous_dataset = train_normalous_dataset.concatenate(val_normalous_dataset)

    train_val_dataset = train_val_cancerous_dataset.concatenate(train_val_normalous_dataset)
    train_dataset = train_cancerous_dataset.concatenate(train_normalous_dataset)
    val_dataset = val_cancerous_dataset.concatenate(val_normalous_dataset)
    test_dataset = test_cancerous_dataset.concatenate(test_normalous_dataset)

    return train_val_dataset, train_dataset, val_dataset, test_dataset


# Upload files to GCP bucket storage
def upload_files_to_gcp(bucket_name: str, source_directory: str):
    """
    Uploads all files from a local directory to a GCP bucket.

    Parameters:
    - bucket_name: Name of the GCP bucket.
    - source_directory: Local directory from which to upload files.

    Return:
    - None
    """

    # Initialize GCP Storage client
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Ensure the destination blob folder path ends with '/'
    if destination_blob_folder and not destination_blob_folder.endswith("/"):
        destination_blob_folder += "/"

    # Walk through the source directory
    for root, dirs, files in os.walk(source_directory):
        for filename in files:
            # Construct the local file path
            local_path = os.path.join(root, filename)
            # print(local_path)

            # Construct the destination path in the bucket
            if destination_blob_folder:
                relative_path = os.path.relpath(local_path, source_directory)
                blob_path = destination_blob_folder + relative_path
            else:
                blob_path = filename

            # Upload the file
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_path)
            print(f'Uploaded {local_path} to "gs://{bucket_name}/{blob_path}"')


def create_and_upload_merged_csv(
    bucket_name: str,
    metadata_csv,
    output_csv_name: str = "ready_to_train.csv",
    file_extension: str = ".jpg",
):
    """
    Fetches files with a specific extension from a GCP bucket, merges their paths with another DataFrame,
    and uploads the merged DataFrame as a CSV to the bucket.

    Parameters:
    - bucket_name: The name of the GCP bucket.
    - metadata_csv: The DataFrame to merge with. It should have columns 'id' and 'label'.
    - output_csv_name: The name of the output CSV file to be stored in the bucket.
    - file_extension: The file extension to filter by. Default is '.jpg'.

    Return:
    - None
    """

    # Initialize a GCP Storage client
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Create a list to hold file information
    files_info = []

    # Iterate over the files in the bucket, filtering by the specified extension
    for blob in bucket.list_blobs():
        if blob.name.lower().endswith(file_extension):
            file_id = blob.name.rsplit(".", 1)[0]  # Extract the file ID
            files_info.append(
                {
                    "image_id": np.int64(int(file_id)),
                    "path": f"gs://{bucket_name}/{blob.name}",
                }
            )

    # Create a DataFrame from the file information
    df_files = pd.DataFrame(files_info)

    # Select only the columnes we need
    metadata_csv = metadata_csv[["image_id", "cancer"]]

    # Merge the DataFrames on the 'id' column
    merged_df = pd.merge(
        df_files, metadata_csv, on="image_id", how="inner"
    )  # Use for the final CSV
    merged_df = pd.merge(
        df_files, metadata_csv, on="image_id", how="left"
    )  # Use for the tests

    # Convert the DataFrame to a CSV string
    csv_string = merged_df.to_csv(index=False)

    # Save the CSV string to a file in the bucket
    blob = bucket.blob(output_csv_name)
    blob.upload_from_string(csv_string, "text/csv")
    print(f'CSV file "{output_csv_name}" uploaded to bucket "{bucket_name}".')


#####
# Dataset Creation
#####
# Load and process images
def load_and_process_image(file_path: str, label):
    """
    Loads and processes an image file for model training.

    Parameters:
    - file_path: The path to the image file.
    - label: The label associated with the image file.

    Returns:
    - Tuple containing the processed image and its label.
    """

    img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(img, channels=1)
    img = tf.image.resize(img, [128, 128])  # Resize images
    img = img / 255.0  # Normalize to [0,1]
    return img, label


def create_dataset(input: str = "local"):
    """
    Creates a dataset for model training.

    Parameters:
    - input: Specifies the source of the dataset, 'local' or 'cloud'.

    Returns:
    - TensorFlow dataset object.
    """
    # local or cloud
    # Load the dataset
    if input == "local":
        df = pd.read_csv(os.path.join(METADATA_DIR, 'ready_to_train.csv'))
    if input == "cloud":
        df = pd.read_csv("gs://mammo_data/ready_to_train.csv")

    # Create a TensorFlow dataset
    paths = df["path"].values
    labels = df["cancer"].values

    labels = tf.cast(labels, dtype=tf.int32)

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(load_and_process_image)

    return dataset


def batch_dataset(dataset, batch_size: int):
    """
    Batches the dataset with the specified batch size.

    Parameters:
    - dataset: The dataset to batch.
    - batch_size: The size of each batch.

    Returns:
    - Batched dataset.
    """
    return dataset.batch(batch_size)


def split_dataset(batched_dataset, ratio: float = 0.8):
    """
    Splits the batched dataset into training and testing datasets.

    Parameters:
    - batched_dataset: The batched dataset to split.

    Returns:
    - Tuple containing the training and testing datasets.
    """
    size = int(len(batched_dataset) * ratio)

    train = batched_dataset.take(size)
    test = batched_dataset.skip(size)

    return train, test
