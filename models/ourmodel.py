"""Classification"""


from functions import save_model
from keras.utils import Sequence
from tensorflow import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from keras.metrics import SparseCategoricalCrossentropy


def OurModel(nb_classes, input_shape, learning_rate=0.001, save_model_png=False):
    # CNN model:
    model = keras.models.Sequential()
    # L1: Conv layer
    model.add(layers.Conv2D(filters=10, kernel_size=(3, 3),
                            strides=(1, 1), padding="same",
                            activation="relu",
                            input_shape=input_shape))  # arg only in L1
    # L2: pooling layer -max pooling-
    model.add(layers.MaxPool2D(pool_size = (2, 2)))
    for i in range(3):
        model.add(layers.Conv2D(filters=16, kernel_size=(3, 3),
                                strides=(1, 1), padding="same",
                                activation="relu"))  # arg only in L1
        # L2: pooling layer -max pooling-
        model.add(layers.MaxPool2D(pool_size = (2, 2)))

    # Conv layer
    model.add(layers.Conv2D(filters=16, kernel_size=(3, 3),
                            strides=(1, 1), padding="same",
                            activation="relu"))
    # L4: pooling layer
    model.add(layers.MaxPool2D(pool_size=(4, 4)))
    # L5: Flatten
    model.add(layers.Flatten())
    # L6: fully connected layer / dense layer
    model.add(layers.Dense(units=64, activation="relu"))
    model.add(layers.Dense(units=32, activation="relu"))
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
        save_model(model, output_path="models/summaries/ourmodel.png")

    return model
