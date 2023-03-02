"""CLASSIFICATION
Predicting feature number 2 (column 1 in features_set001.csv)
giving 20000 images using convolutional neural network

- predir les variables homogenes = 3 1eres columns in set 001
- objectif: proposer un model et l'executer
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import nn

from functions import plot_loss, into_classes
from set001_feat2_cnn_model import model


# Load trining and test data
dataset = np.load('database/feature_set001_feat2.npz')
train_images, train_targets = dataset["train_images"], dataset["train_targets"]
test_images, test_targets = dataset["test_images"], dataset["test_targets"]

# limit training data for computation issues (to delete)
n_train = 18000
n_test = 500
train_images, train_targets = train_images[: n_train], train_targets[: n_train]
test_images, test_targets = test_images[: n_test], test_targets[: n_test]

# converting target values into classes
nb_classes = 10
train_targets = into_classes(feature_vec=train_targets, nb_classes=nb_classes)
test_targets = into_classes(feature_vec=test_targets, nb_classes=nb_classes)


# IV/ training parameters
batch_size = 20
epochs = 5


# V/ Training
train_history = model.fit(x=train_images, y=train_targets,
                batch_size=batch_size, epochs=epochs,
                shuffle=False, verbose=2,
                validation_split = 0.2)  # use 20% for valid automatically

plot_loss(train_history, savepath="figures/set001_feat2_model_history.png")


# VI/ Evaluation: evaluation of the model on test set
model.evaluate(x=test_images, y=test_targets,
               batch_size=batch_size,
               verbose=2)


# VII/ Predict: chose an exmple and predict its labels

# predict class: method 2
predictions = model.predict(test_images, batch_size=batch_size)
predictions = nn.softmax(predictions)

pred0, label0 = predictions[0], np.argmax(predictions[0])
print("pred0 = {} \n \n label0 = {} \n".format(pred0, label0))
