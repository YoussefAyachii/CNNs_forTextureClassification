"""set001_feat1_cnn model: CNN model to predict feature
number 0 (column 0 in features_set001.csv) giving 20000 images
using convolutional neural network.
model de regression"""


from functions import getsets
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
dataset = np.load('database/feature_set001_feat1.npz')
train_images, train_targets = dataset["train_images"], dataset["train_targets"]
test_images, test_targets = dataset["test_images"], dataset["test_targets"]


# CNN model:
model = keras.models.Sequential()

# L1: Conv layer
model.add(layers.Conv2D(filters=96, kernel_size=(7, 7),
                        strides=(2, 2), padding="valid",
                        activation="relu",
                        input_shape=train_images[0].shape))  # arg only in L1
# L2: pooling layer -max pooling-
model.add(layers.MaxPool2D(pool_size = (3, 3)))

# Conv layer
model.add(layers.Conv2D(filters=256, kernel_size=(5, 5),
                        strides=(2, 2), padding="valid",
                        activation="relu"))  # arg only in L1
# pooling layer -max pooling-
model.add(layers.MaxPool2D(pool_size = (3, 3)))

# Conv layer
model.add(layers.Conv2D(filters=384, kernel_size=(3, 3),
                        strides=(2, 2), padding="valid",
                        activation="relu"))  # arg only in L1
# pooling layer -max pooling-
model.add(layers.MaxPool2D(pool_size = (3, 3)))

# Conv layer
model.add(layers.Conv2D(filters=384, kernel_size=(3, 3),
                        strides=(2, 2), padding="valid",
                        activation="relu"))  # arg only in L1
# pooling layer -max pooling-
model.add(layers.MaxPool2D(pool_size = (3, 3)))

# Conv layer
model.add(layers.Conv2D(filters=256, kernel_size=(3, 3),
                        strides=(2, 2), padding="valid",
                        activation="relu"))  # arg only in L1
# pooling layer -max pooling-
model.add(layers.MaxPool2D(pool_size = (3, 3)))


# L5: Flatten
model.add(layers.Flatten())
# L6: fully connected layer / dense layer
model.add(layers.Dense(units=512, activation="relu"))
model.add(layers.Dense(units=256, activation="relu"))
model.add(layers.Dense(units=1))
print(model.summary())

# loss
loss = keras.losses.MeanSquaredError()
# optimizer (learning rate (lr) = hyper parameter to fix by CV)
optim = keras.optimizers.Adam(learning_rate=0.1)
# metrics: mean absolute error
metrics = ["mae", rmse()]



# III/ compilation: configure the model for training
model.compile(loss=loss, optimizer=optim, metrics=metrics)
