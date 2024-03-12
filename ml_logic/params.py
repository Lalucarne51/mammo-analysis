import os

##################  VARIABLES  ##################

MODEL_TARGET = os.environ.get("MODEL_TARGET")
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_REGION = os.environ.get("GCP_REGION")

BUCKET_NAME = os.environ.get("BUCKET_NAME")
BUCKET_MODEL = os.environ.get("BUCKET_MODEL")
INSTANCE = os.environ.get("INSTANCE")


##################  CONSTANTS  #####################
CUR_DIR = os.getcwd()
LOCAL_REGISTRY_PATH = os.path.join(CUR_DIR, "registry")
if not os.path.exists(LOCAL_REGISTRY_PATH):
    os.makedirs(LOCAL_REGISTRY_PATH)


EPOCHS = 100
BATCH_SIZE = 64
DIM = 128
