import os

##################  VARIABLES  ##################

MODEL_TARGET = os.environ.get("MODEL_TARGET")
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_REGION = os.environ.get("GCP_REGION")

BUCKET_NAME = os.environ.get("BUCKET_NAME")
INSTANCE = os.environ.get("INSTANCE")


##################  CONSTANTS  #####################
LOCAL_REGISTRY_PATH = os.path.join(
    os.path.expanduser("~"),
    "code",
    "Lalucarne51",
    "mammo-analysis",
)

EPOCHS = 100
BATCH_SIZE = 64
DIM = 128
