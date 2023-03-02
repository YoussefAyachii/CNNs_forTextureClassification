"""Testing the conversion of the dataset of .jpg images
into .npz unique file"""

import numpy as np

dataset = np.load('database/feature_set001_feat0.npz')
train_images, train_targets = dataset["train_images"], dataset["train_targets"]
test_images, test_targets = dataset["test_images"], dataset["test_targets"]

print(train_images.shape)
print(test_images.shape)

print(train_images[0].shape)
print(test_images[0].shape)