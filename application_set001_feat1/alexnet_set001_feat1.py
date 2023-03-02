"""Predicting feature number 0 (column 1 in features_set001.csv)
giving 20000 images using convolutional neural network

- predir les variables homogenes = 3 1eres columns in set 001
- objectif: proposer un model et l'executer
"""

import numpy as np
import matplotlib.pyplot as plt

from tensorflow import nn
from functions import plot_loss, into_classes, load_train_test_to_classes
from models.alexnet import AlexNet



# dataset specification and hyperparameters selection
set_nb = 1
feat_nb = 1

nb_classes = 5
class_names = ["no isotropic", "low isotropic", "isotropic",
               "high isotropic", "very high isotropic"]

n_train = 2000
n_test = 200
range_values = (0, 3.14)

input_shape = (64, 64, 1)
learning_rate = 1e-6
batch_size = 32
epochs = 3


# load data
train_images , train_targets, test_images, test_targets = load_train_test_to_classes(
    set_nb=set_nb, feat_nb=feat_nb, nb_classes=nb_classes,
    range_values=range_values, n_train=n_train, n_test=n_test, img_output="gray")



# Define model
model = AlexNet(nb_classes, input_shape, learning_rate, save_model_png=False)


# V/ Training
train_history = model.fit(x=train_images, y=train_targets,
                batch_size=batch_size, epochs=epochs,
                shuffle=True, verbose=2,
                validation_split = 0.2)  # use 20% for valid automatically

plot_loss(train_history, savepath="figures/alexnet_set001_feat1.png")


# VI/ Evaluation: evaluation of the model on test set
model.evaluate(x=test_images, y=test_targets,
               batch_size=batch_size,
               verbose=2)


# VII/ Predict: chose an exmple and predict its labels
predictions = model.predict(test_images, batch_size=batch_size)
predictions = nn.softmax(predictions)

label010 = np.argmax(predictions[0:10], axis=1)
print("\n predicted labels 0:10 = {} \n".format(label010))

print("correct labels 0:10 = {} \n".format(test_targets[0:10]))

# print("verify sum probabilities=1 : ", np.sum(pred05[0]))
print("accuracy on test set : ", np.mean(np.argmax(predictions, axis=1) == test_targets))
