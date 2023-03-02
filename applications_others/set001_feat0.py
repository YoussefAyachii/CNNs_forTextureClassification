"""Predicting feature number 0 (column 1 in features_set001.csv)
giving 20000 images using convolutional neural network

- predir les variables homogenes = 3 1eres columns in set 001
- objectif: proposer un model et l'executer
"""

import numpy as np
import matplotlib.pyplot as plt

from functions import plot_loss
from set001_feat0_cnn_model import model


# Load trining and test data
dataset = np.load('database/feature_set001_feat0.npz')
train_images, train_targets = dataset["train_images"], dataset["train_targets"]
test_images, test_targets = dataset["test_images"], dataset["test_targets"]

# limit training data for computation issues (to delete)
n_train = 18000
n_test = 2000
train_images, train_targets = train_images[: n_train], train_targets[: n_train]
test_images, test_targets = test_images[: n_test], test_targets[: n_test]

# data exploration
print("train_images.shape \n ", train_images.shape)  # must be (64, 64, 1) instead of (64, 64)
print("train_targets.shape \n ", train_targets.shape)

print("test_images.shape \n ", test_images.shape)
print("test_targets.shape \n ", test_targets.shape)

print("train_targets \n ", train_targets[0:10])
print("test_targets \n ", test_targets[0:10])

# IV/ training parameters
batch_size = 20
epochs = 5


# V/ Training
train_history = model.fit(x=train_images, y=train_targets,
                batch_size=batch_size, epochs=epochs,
                shuffle=True, verbose=2,
                validation_split = 0.2)  # use 20% for valid automatically

plot_loss(train_history, savepath="figures/set001_feat0_model_history.png")


# VI/ Evaluation: evaluation of the model on test set
model.evaluate(x=test_images, y=test_targets,
               batch_size=batch_size,
               verbose=2)


# VII/ Predict: chose an exmple and predict its labels

predictions = model.predict(test_images)
print("target of test_images[0:10]: \n", predictions[0: 10])
print("mse targt vs predictions : \n", np.mean(np.square(predictions-test_targets)))
