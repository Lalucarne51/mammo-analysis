import numpy as np
import time

from tensorflow import keras
from keras import Model, Sequential, layers, regularizers, optimizers
from keras.callbacks import EarlyStopping


"""
def plot_loss_accuracy(history, title=None):
    fig, ax = plt.subplots(1,2, figsize=(20,7))

    # --- LOSS ---

    ax[0].plot(history.history['loss'])
    ax[0].plot(history.history['val_loss'])
    ax[0].set_title('Model loss')
    ax[0].set_ylabel('Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylim((0,3))
    ax[0].legend(['Train', 'Test'], loc='best')
    ax[0].grid(axis="x",linewidth=0.5)
    ax[0].grid(axis="y",linewidth=0.5)

    # --- ACCURACY

    ax[1].plot(history.history['accuracy'])
    ax[1].plot(history.history['val_accuracy'])
    ax[1].set_title('Model Accuracy')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_xlabel('Epoch')
    ax[1].legend(['Train', 'Test'], loc='best')
    ax[1].set_ylim((0,1))
    ax[1].grid(axis="x",linewidth=0.5)
    ax[1].grid(axis="y",linewidth=0.5)

    if title:
        fig.suptitle(title)"""


'''PREPROCESSING

### Normalizing pixels' intensities
X_train = images_train / 255.
X_train_small = images_train_small / 255.
X_test = images_test / 255.
X_test_small = images_test_small / 255.

### Encoding the labels
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(labels_train, 10)
y_train_small = to_categorical(labels_train_small, 10)
y_test = to_categorical(labels_test, 10)
y_test_small = to_categorical(labels_test_small, 10)


'''
'''
IMAGE GENERATOR

from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    featurewise_center = False,
    featurewise_std_normalization = False,
    rotation_range = 10,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    horizontal_flip = True,
    zoom_range = (0.8, 1.2),
    )

datagen.fit(X_train)
datagen

X_augmented_iterator = datagen.flow(X_train, shuffle=False, batch_size=1)
X_augmented_iterator


'''



def initialize_model(input_shape: tuple) -> Model:

    #############################
    #  1 - Model architecture   #
    #############################

    ## layers.Reshape((28, 28, 1), input_shape=(28,28)) ##


    model = Sequential()

    model.add(layers.Conv2D(128, 5, activation='relu', padding='valid', input_shape=(256, 256, 1)))
    model.add(layers.Conv2D(128, 5, strides=(2,2), activation='relu', padding='valid'))
    model.add(layers.MaxPooling2D(2, padding='valid'))
    model.add(layers.Conv2D(256, 5, activation='relu', padding='same'))
    model.add(layers.Conv2D(256, 5, strides=(2,2), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Conv2D(256, 3, activation='relu', padding='same'))
    model.add(layers.Conv2D(256, 3, activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(1, activation='sigmoid'))

    return model




def compile(model: Model, learning_rate=0.0005):
    #############################
    #  2 - Optimization Method  #
    #############################


    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['recall', 'accuracy', 'precision'])

    return model

"""
model = initialize_model()



# STEP 2: OPTIMIZATION METHODS
model.compile(loss='categorical_crossentropy', optimizer='adam')

# SETP 3: DATA AND FITTING METHODS
model.fit(X, y, batch_size=32, epochs=100)


history = model_compile.fit(X_train, y_train,
            batch_size=16,
          epochs=500,
          validation_split=0.3,
          callbacks=[es])
"""
def train_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=256,
        patience=2,
        validation_data=None, # overrides validation_split
        validation_split=0.2
    ) -> Tuple[Model, dict]:
    """
    Fit the model and return a tuple (fitted_model, history)
    """

    es = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X,
        y,
        validation_data=validation_data,
        validation_split=validation_split,
        epochs=100,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0
    )

    print(f"✅ Model trained on {len(X)} rows with min val MAE: {round(np.min(history.history['val_mae']), 2)}")

    return model, history


def evaluate_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=64
    ) -> Tuple[Model, dict]:
    """
    Evaluate trained model performance on the dataset
    """

    print(Fore.BLUE + f"\nEvaluating model on {len(X)} rows..." + Style.RESET_ALL)

    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    metrics = model.evaluate(
        x=X,
        y=y,
        batch_size=batch_size,
        verbose=0,
        # callbacks=None,
        return_dict=True
    )

    loss = metrics["loss"]
    mae = metrics["mae"]

    print(f"✅ Model evaluated, MAE: {round(mae, 2)}")

    return metrics
