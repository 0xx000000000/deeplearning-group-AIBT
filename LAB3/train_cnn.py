"""LAB3 - Question 8: train the final CIFAR-10 CNN and save it.

Takes no input. Loads CIFAR-10, trains the CNN on the TRAINING data only
(the test set is never touched), then saves the model as
SO_HOEGY_ANDRIAMANAMIRIJA.keras next to this script.

Usage:
    python train_cnn.py
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
EPOCHS = 30
BATCH_SIZE = 64
VAL_SIZE = 5000  # held out from the TRAINING set to monitor training
MODEL_NAME = "SO_HOEGY_ANDRIAMANAMIRIJA.keras"


def conv_block(x, filters, dropout):
    """Two 3x3 convolutions (each followed by batch norm + ReLU), then pooling and dropout."""
    for _ in range(2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    return layers.Dropout(dropout)(x)


def build_model():
    inputs = keras.Input(shape=(32, 32, 3))
    # Data augmentation: only active during training, identity at inference.
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomTranslation(0.1, 0.1, fill_mode="reflect")(x)

    x = conv_block(x, 32, 0.2)   # 32x32 -> 16x16
    x = conv_block(x, 64, 0.3)   # 16x16 -> 8x8
    x = conv_block(x, 128, 0.4)  # 8x8   -> 4x4

    x = layers.Flatten()(x)
    x = layers.Dense(128, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="cifar10_cnn")
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def callbacks():
    return [
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    ]


def main():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    (x_train, y_train), _ = keras.datasets.cifar10.load_data()  # test split deliberately ignored
    x_train = x_train.astype("float32") / 255.0
    y_train = keras.utils.to_categorical(y_train, 10)

    x_tr, y_tr = x_train[:-VAL_SIZE], y_train[:-VAL_SIZE]
    x_val, y_val = x_train[-VAL_SIZE:], y_train[-VAL_SIZE:]

    model = build_model()
    model.summary()
    model.fit(x_tr, y_tr, validation_data=(x_val, y_val),
              epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks(), verbose=2)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_NAME)
    model.save(out)
    print("Model saved to", out)


if __name__ == "__main__":
    main()
