"""set001_feat0_cnn model: CNN model to predict feature
number 0 (column 0 in features_set001.csv) giving 20000 images
using convolutional neural network"""


from functions import getsets
from os import listdir
from os.path import isfile, join
from keras.utils import Sequence
from tensorflow import keras
from keras import layers
from imageio.v2 import imread
import numpy as np
import matplotlib.pyplot as plt
from keras.metrics import RootMeanSquaredError as mse


# get train_images

# Load trining and test data
dataset = np.load('database/feature_set002_feat1.npz')
train_images, train_targets = dataset["train_images"], dataset["train_targets"]
test_images, test_targets = dataset["test_images"], dataset["test_targets"]


# CNN model:
model = keras.models.Sequential()
# L1: Conv layer
model.add(layers.Conv2D(filters=4, kernel_size=(3, 3),
                        strides=(1, 1), padding="same",
                        activation="relu",
                        input_shape=train_images[0].shape))  # arg only in L1
# L2: pooling layer -max pooling-
model.add(layers.MaxPool2D(pool_size = (2, 2)))
# Conv layer
model.add(layers.Conv2D(filters=16, kernel_size=(3, 3),
                        strides=(1, 1), padding="valid",
                        activation="relu"))  # arg only in L1
# L2: pooling layer -max pooling-
model.add(layers.MaxPool2D(pool_size = (2, 2)))
# Flatten
model.add(layers.Flatten())
# L6: fully connected layer / dense layer
model.add(layers.Dense(units=64, activation="relu"))
model.add(layers.Dense(units=1))
print(model.summary())

# loss
loss = keras.losses.MeanSquaredError()
# optimizer (learning rate (lr) = hyper parameter to fix by CV)
optim = keras.optimizers.Adam(learning_rate=0.01)
# metrics: mean absolute error
metrics = [mse()]



# III/ compilation: configure the model for training
model.compile(loss=loss, optimizer=optim, metrics=metrics)
