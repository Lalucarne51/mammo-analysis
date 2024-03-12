import numpy as np
import time

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from params import *


#####
# Model
#####
def initialize_model():
    """
    Initializes a sequential model for binary classification.

    Returns:
    - TensorFlow Sequential model.
    """
    model = Sequential()

    # Trainable params 444_203
    model.add(Conv2D(16, (4, 4), input_shape=(DIM, DIM, 1), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=50, activation="relu"))
    model.add(Dense(units=10, activation="relu"))
    model.add(Dense(units=1, activation="sigmoid"))

    return model


#####
# Callback
#####
# class custom_callback(tf.keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs={}):
#         if logs["accuracy"] >= 0.97:
#             self.model.stop_training = True

# custom_callback = custom_callback()

timestamp = time.strftime("%Y%m%d-%H%M%S")

checkpoint_path = os.path.join(LOCAL_REGISTRY_PATH, "checkpoints", f"{timestamp}.json")
checkpoint = ModelCheckpoint(
    checkpoint_path,
    monitor="val_recall",
    verbose=1,
    save_weights_only=True,
    mode="max",
    save_best_only=True,
)
es = EarlyStopping(monitor="val_recall", patience=10)
callbacks_list = [checkpoint, es]

# model.load_weights("FILENAME_PATH")

#####
# Optimizer
#####
# optimizer = tf.keras.optimizers.legacy.Adam(
#     learning_rate=0.000001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Adam"
# )
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.000001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Adam"
)

#####
# Loss Fn
#####
lossfn = tf.keras.losses.BinaryCrossentropy(
    from_logits=False, label_smoothing=0.0, axis=-1, name="binary_crossentropy"
)


#####
# Workflow
#####
def initialize_and_compile_model(optimizer, lossfn):
    """
    Initializes the model, compiles it with the specified optimizer and loss function,
    and prints the model summary.

    Parameters:
    - optimizer: The optimizer to use for training the model.
    - lossfn: The loss function to use for training.

    Returns:
    - Compiled TensorFlow model.
    """
    print("\nInit the model :")
    model = initialize_model()
    model.compile(
        optimizer=optimizer, loss=lossfn, metrics=["accuracy", "Recall", "Precision"]
    )
    return model


def train_model(model, train, test, epochs: int, callbacks: list):
    """
    Trains the model on the training dataset and validates it on the testing dataset.

    Parameters:
    - model: The model to train.
    - train: The training dataset.
    - test: The testing dataset.
    - epochs: The number of epochs to train for.
    - callbacks: A list of callbacks to use during training.

    Returns:
    - History object resulting from model training.
    """
    history = model.fit(
        train,
        validation_data=test,
        epochs=epochs,
        callbacks=callbacks,
    )
    return history


# def evaluate_model(model: Model,
#                    dataset,
#                    batch_size=64):
#     """
#     Evaluate trained model performance on the dataset
#     """

#     print(Fore.BLUE + f"\nEvaluating model on {len(X)} rows..." + Style.RESET_ALL)

#     if model is None:
#         print(f"\n❌ No model to evaluate")
#         return None

#     metrics = model.evaluate(
#         dataset,
#         batch_size=batch_size,
#         verbose=0,
#         # callbacks=None,
#         return_dict=True
#     )

#     loss = metrics["loss"]
#     recall = metrics["recall"]

#     print(f"✅ Model evaluated, recall: {round(recall, 2)}")

#     return metrics
