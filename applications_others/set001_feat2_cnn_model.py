"""CLASSIFICATION
set001_feat2_cnn model: CNN model to predict feature
number 0 (column 0 in features_set001.csv) giving 20000 images
using convolutional neural network.
model de CLASSIFICATION"""


from functions import getsets, into_classes
from os import listdir
from os.path import isfile, join
from keras.utils import Sequence
from tensorflow import keras
from keras import layers
from imageio.v2 import imread
import numpy as np
import matplotlib.pyplot as plt
from keras.metrics import RootMeanSquaredError as rmse


# get train_images

# Load trining and test data
dataset = np.load('database/feature_set001_feat2.npz')
train_images, train_targets = dataset["train_images"], dataset["train_targets"]
test_images, test_targets = dataset["test_images"], dataset["test_targets"]

# nb classes
nb_classes = 10

# CNN model:
model = keras.models.Sequential()
# L1: Conv layer
model.add(layers.Conv2D(filters=8, kernel_size=(3, 3),
                        strides=2, padding="same",
                        activation="relu",
                        input_shape=train_images[0].shape))  # arg only in L1
# L2: pooling layer -max pooling-
model.add(layers.MaxPool2D(pool_size = (2, 2), strides=2))
# L3: Conv layer
model.add(layers.Conv2D(filters=16, kernel_size=(3, 3),
                        strides=(1, 1), padding="valid",
                        activation="relu"))
# L4: pooling layer
model.add(layers.MaxPool2D(pool_size=(2, 2), strides=2))
# L5: Flatten
model.add(layers.Flatten())
# L6: fully connected layer / dense layer
#model.add(layers.Dense(units=120, activation="relu"))
model.add(layers.Dense(units=64, activation="relu"))
model.add(layers.Dense(units=nb_classes, activation="softmax"))
print(model.summary())


# loss
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
# optimizer (learning rate (lr) = hyper parameter to fix by CV)
optim = keras.optimizers.Adam(learning_rate=0.001)
# metrics: mean absolute error
metrics = ["accuracy"]


# III/ compilation: configure the model for training
model.compile(loss=loss, optimizer=optim, metrics=metrics)
