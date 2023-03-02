"""Convert list of png images to a single .npz file"""

from functions import getarray, allfilenames, getsets
import numpy as np


# load features set
features_set001 = np.genfromtxt('database/features_set001.csv', delimiter=',')
features_set002 = np.genfromtxt('database/features_set002.csv', delimiter=',')


# Save training set and test set with target = to chosen feature
feat = [0, 1, 2, 3]
for i, f in enumerate(feat):
    sets = getsets(n_train=18000, n_test=2000, featuretab=features_set001,
                   feature_nb=f, dir_path="database/images_set001/",
                   expected=20000, img_shape=(64, 64),
                   npz_dir="database/feature_set001")


feat = [0, 1, 2, 3]
for i, f in enumerate(feat):
    sets = getsets(n_train=18000, n_test=2000, featuretab=features_set002,
                   feature_nb=f, dir_path="database/images_set002/",
                   expected=20000, img_shape=(64, 64),
                   npz_dir="database/feature_set002")
