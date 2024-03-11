import numpy as np
import time

from colorama import Fore, Style
from typing import Tuple

# Timing the TF import
print(Fore.BLUE + "\nLoading TensorFlow..." + Style.RESET_ALL)
start = time.perf_counter()

from tensorflow import keras
from keras import Model, Sequential, layers, regularizers, optimizers
from keras.callbacks import EarlyStopping

def initialize_model(input_shape):
    """
    Initialize the Neural Network with random weights
    """

    model = models.Sequential()
    model.add(layers.Conv2D(16, (4, 4), input_shape = (128, 128, 1), activation = 'relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation = 'relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(units = 10, activation = 'relu'))
    model.add(layers.Dense(units = 1, activation = 'sigmoid'))
    model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics =['recall'])

    print("✅ Model initialized")

    return model

def compile_model(model: Model):
    """
    Compile the Neural Network
    """
    optimizer = optimizers.Adam(learning_rate=0.0005)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["recall, Accuracy, Precision"])

    print("✅ Model compiled")

    return model

def train_model(model,
                dataset,
                batch_size=256,
                patience=2,
                validation_data=None,
                validation_split=0.2):
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    es = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        dataset,
        batch_size=batch_size,
        epochs=100,
        patience=patience,
        validation_data=validation_data,
        validation_split=validation_split,
        callbacks=[es],
        verbose=0
        )

    print(f"✅ Model trained with min val recall: {round(np.min(history.history['val_recall']), 2)}")

    return model, history

def evaluate_model(model: Model,
                   dataset,
                   batch_size=64):
    """
    Evaluate trained model performance on the dataset
    """

    print(Fore.BLUE + f"\nEvaluating model on {len(X)} rows..." + Style.RESET_ALL)

    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    metrics = model.evaluate(
        dataset,
        batch_size=batch_size,
        verbose=0,
        # callbacks=None,
        return_dict=True
    )

    loss = metrics["loss"]
    recall = metrics["recall"]

    print(f"✅ Model evaluated, recall: {round(recall, 2)}")

    return metrics
