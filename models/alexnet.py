"""AlexNet classification"""


from functions import save_model
from os import listdir
from os.path import isfile, join
from keras.utils import Sequence
from tensorflow import keras
from keras import layers
from imageio.v2 import imread
import numpy as np
import matplotlib.pyplot as plt
from keras.metrics import SparseCategoricalCrossentropy


def AlexNet(nb_classes, input_shape, learning_rate=0.001, save_model_png=False):
    # CNN model:
    model = keras.models.Sequential()
    # L1: Conv layer
    model.add(layers.Conv2D(filters=96, kernel_size=11,
                            strides=4, padding="valid",
                            activation="relu",
                            input_shape=input_shape))
    model.add(layers.MaxPool2D(pool_size = (3, 3),
                            strides=2))
    
    # Conv with padding 2
    model.add(keras.layers.ZeroPadding2D(padding=(2, 2)))
    model.add(layers.Conv2D(filters=256, kernel_size=5,
                            strides=1, padding="same",
                            activation="relu"))
    # pooling layer -max pooling-
    model.add(layers.MaxPool2D(pool_size = (3, 3),
                               strides=2))


    model.add(layers.Conv2D(filters=384, kernel_size=3,
                            strides=1, padding="same",
                            activation="relu"))  # arg only in L1

    model.add(layers.Conv2D(filters=384, kernel_size=3,
                            strides=1, padding="same",
                            activation="relu"))

    model.add(layers.Conv2D(filters=256, kernel_size=3,
                            strides=1, padding="same",
                            activation="relu"))

    # Pooling
    model.add(layers.MaxPool2D(pool_size = (3, 3),
                            strides=2))
    # Flatten
    model.add(layers.Flatten())
    # Dropout 1
    keras.layers.Dropout(rate=0.5)
    # Dense 1
    model.add(layers.Dense(units= 4096, activation="relu"))
    # Dropout 2
    keras.layers.Dropout(rate=0.5)
    # Dense 2
    model.add(layers.Dense(units= 4096, activation="relu"))
    # Dense 3
    model.add(layers.Dense(units= 1000, activation="relu"))
    # Dense 3
    model.add(layers.Dense(units=nb_classes, activation="softmax"))

    print(model.summary())

    # loss
    loss = keras.losses.SparseCategoricalCrossentropy()
    # optimizer (learning rate (lr) = hyper parameter to fix by CV)
    optim = keras.optimizers.Adam(learning_rate=learning_rate)
    # metrics: mean absolute error
    metrics = ["accuracy"]


    # III/ compilation: configure the model for training
    model.compile(loss=loss, optimizer=optim, metrics=metrics)

    # save model into ong
    if save_model_png != False:
        save_model(model, output_path="models/summaries/alexnet.png")

    return model
