import numpy as np
# import time

# from colorama import Fore, Style
# from typing import Tuple

### Timing the TF import
# print(Fore.BLUE + "\nLoading TensorFlow..." + Style.RESET_ALL)
# start = time.perf_counter()

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

#####
# Model
#####
def initialize_model():
    """
    Initializes a sequential model for binary classification.

    Returns:
    - TensorFlow Sequential model.
    """
    model = model.Sequential()

    model.add(layers.Conv2D(64, 7, activation='relu', padding='valid', input_shape=(256, 256, 1)))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Conv2D(128, 3, activation='relu', padding='same'))
    model.add(layers.Conv2D(128, 3, activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Conv2D(256, 3, activation='relu', padding='same'))
    model.add(layers.Conv2D(256, 3, activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(10, activation='relu'))

    model.add(layers.Dense(1, activation='sigmoid'))

    return model

    return model


#####
# Callback
#####
class custom_callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs["accuracy"] >= 0.97:
            self.model.stop_training = True


custom_callback = custom_callback()

#####
# Optimizer
#####
optimizer = tf.keras.optimizers.legacy.Adam(
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
    model.compile(optimizer=optimizer, loss=lossfn, metrics=["accuracy"])
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
