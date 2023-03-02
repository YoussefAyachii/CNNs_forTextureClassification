"""functions"""


from os import listdir
from os.path import isfile, join
from keras.utils import Sequence
from tensorflow import keras
from keras import layers
from imageio.v2 import imread
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dataframe_image as dfi



# load data

def imagepath(input_dir, image_name):
    """return the path to a given image example given its name and dir."""
    return(input_dir + image_name)


def getarray(input_dir, image_name):
    """return the image.png as an np.array given its name and dir"""
    image_path = imagepath(input_dir, image_name)
    array = imread(image_path)
    return np.asarray(array)


def allfilenames(dir_path, expected=False):
    """return all file names in a given dir
    verification can be done by specifying
    the total number of images expected."""

    # ignore hidden files (ex. ".DS_store")
    filenames = [f for f in listdir(dir_path) if not f.startswith('.')]
    filenames.sort()
    
    if expected != False:
        assert expected == len(filenames)
    return filenames


# convert feature column (continue variable) into classes (discrete)

def into_classes(feature_vec, nb_classes=20, return_classes=False, range_values=(0, 1)):
    """convert feature column (continue variable) into
    classes (discrete)"""    

    # check args
    assert len(feature_vec.shape) == 1

    min_val = range_values[0]
    max_val = range_values[1]
    classes = np.linspace(min_val, max_val, nb_classes+1)
    classes_list = np.empty((len(classes)-1, 0)).tolist()
    classes_verif = classes_list
    
    print("classes  = \n ", classes)
    print("classes  = \n ", classes_list)

    for i in range(len(classes)-1):
        classe_inf = classes[i]  # borne inf 
        classe_sup = classes[i + 1]  # borne sup
        
        bool_inf = feature_vec >= classe_inf 
        bool_sup = feature_vec < classe_sup
        bool_inclass = np.logical_and(bool_inf, bool_sup)  # True only if in [borne inf, borne supr[

        # indices of feature_vec that belong to current class
        classe_tmp_indices = [j for j, x in enumerate(bool_inclass) if x]
        if len(classe_tmp_indices) > 0:
            classes_list[i] = classe_tmp_indices

    # return label of the class (1, 2, 3 ..) instead of real value
    classes_names = np.arange(0, len(classes_list))
    feature_vec_discrete = feature_vec
    for k, current_class in enumerate(classes_list):
        # change feature_vec (continuous) to classes (discrete)
        nb_elements_in_current_class = len(current_class)
        feature_vec_discrete[current_class] = classes_names[k]
    feature_vec_discrete.astype(int)
    

    # insure same length original feature vec and output
    assert len(feature_vec) == len(feature_vec_discrete)
    assert len(classes_list) == len(classes_verif)
    
    #concat_list = [j for i in classes_list for j in i]
    #assert len(concat_list) == len(feature_vec)


    if return_classes == True:
        return feature_vec, classes_names
    else:
        return feature_vec


# get train set and test set

# trainset and testset : feature_nb = column number to guess among featurs.csv columns
def getsets(n_train, n_test, featuretab, feature_nb,
            dir_path, expected=20000, img_shape=(64, 64), npz_dir=False):
    """return the training and test sets as numpy arrays."""
    
    # check arguments
    assert n_train + n_test <= expected
    assert feature_nb in np.arange(0, 4, 1)  # features.csv has 4 columns
    assert len(img_shape) == 2
    
    # size of one image
    h, w = img_shape
    # initialize array
    n_tot = n_train + n_test
    z = np.zeros((n_tot, h, w, 1))
    # filnames of all images
    filenames = allfilenames(dir_path, expected)

    for i, filename in enumerate(filenames[:n_tot]):  # prends les 500premiers images
        array_tmp = getarray(input_dir=dir_path, image_name=filename)
        z[i] = array_tmp[:, :, np.newaxis] # (64, 64, 1) instead of (64, 64)

    train_images, train_targets = z[:n_train], featuretab[:n_train, feature_nb]
    test_images, test_targets = z[n_train:n_tot], featuretab[n_train:n_tot, feature_nb]
    
    if npz_dir != False :
        filename = npz_dir + "_feat" + str(feature_nb)
        np.savez(filename,
                 train_images=train_images,
                 train_targets=train_targets,
                 test_images=test_images,
                 test_targets=test_targets)
    else:
        return train_images, train_targets, test_images, test_targets


# to gey scale or rgb (3d) image
def to_grey_or_rgb(img_set, img_output="gray"):
    max_value = np.max(np.ndarray.flatten(img_set))
    gray_img = np.divide(img_set, max_value)
    if img_output != "original":
        if img_output == "gray":
            return gray_img
        elif img_output == "rgb":
            rgb_img = gray_img * 255
            return rgb_img


# load training and test sets with target values (labels) as classes

def load_train_test_to_classes(set_nb, feat_nb, nb_classes,
                               range_values, n_train=2000, n_test=200,
                               img_output="gray", dim="1",
                               return_classes=False):
    """load dataset (images and target column from features.csv)
    and convert the target column to classes"""

    # chack args
    assert set_nb in (1, 2)
    assert feat_nb in (0, 1, 2, 3)
    assert len(range_values) == 2

    # Load trining and test data
    dataset_name = "feature_set00" + str(set_nb) + "_feat" + str(feat_nb)
    dataset = np.load("database/" + dataset_name + ".npz")
    train_images, train_targets = dataset["train_images"], dataset["train_targets"]
    test_images, test_targets = dataset["test_images"], dataset["test_targets"]
    
    train_images = to_grey_or_rgb(test_images, img_output)
    test_images = to_grey_or_rgb(test_images, img_output)

    # converting target values into classes

    train_targets = into_classes(feature_vec=train_targets,
                                nb_classes=nb_classes,
                                return_classes=False,
                                range_values=range_values)

    test_targets = into_classes(feature_vec=test_targets,
                                nb_classes=5,
                                return_classes=return_classes,
                                range_values=range_values)


    print("train_targets : \n", train_targets[0:20])
    print("test_targets : \n", test_targets[0:20])


    # limit training data for computation issues (to delete)
    assert n_train + n_test <= 20000
    n_train = 2000
    n_test = 200
    train_images, train_targets = train_images[: n_train], train_targets[: n_train]
    test_images, test_targets = test_images[: n_test], test_targets[: n_test]

    # from (x, x, 1) to (x, x, 3) for some model issues (ex. resnet50)
    if dim == 3:
        train_images = np.array([np.repeat(img, repeats = 3, axis = -1) for img in train_images])
        test_images = np.array([np.repeat(img, repeats = 3, axis = -1) for img in test_images])
    
    return train_images, train_targets, test_images, test_targets


# visualize train and validation loss values in each iteration
def plot_loss(history, savepath):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.yscale("log")
    plt.xlabel('Epoch')
    plt.ylabel('loss (mse)')
    plt.legend()
    plt.grid(True)
    plt.savefig(savepath)


# save model architecture
def save_model(model, output_path):
    table=pd.DataFrame(columns=["Name","Type","Shape"])
    for layer in model.layers:
        table = table.append(
            {"Name":layer.name,
             "Type": layer.__class__.__name__,
             "Shape":layer.output_shape},
            ignore_index=True)
    #df_styled = table.style.background_gradient()
    dfi.export(table, output_path, max_rows=-1)


    