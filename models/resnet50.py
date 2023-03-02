import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

from tensorflow import keras, nn
from functions import save_model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

def ResNet50_(nb_classes, train_images, learning_rate=0.001, save_model_png=False):
    # Preprocessing
    train_images = preprocess_input(train_images)

    # Define model
    base_model = ResNet50(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # add a fully-connected layer
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    # and a logistic layer (nb classes)
    predictions = tf.keras.layers.Dense(nb_classes, activation='softmax')(x) 
    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)


    # loss
    loss = keras.losses.SparseCategoricalCrossentropy()
    # optimizer (learning rate (lr) = hyper parameter to fix by CV)
    optim = keras.optimizers.Adam(learning_rate=learning_rate)
    # metrics: mean absolute error
    metrics = ["accuracy"]


    # III/ compilation: configure the model for training
    model.compile(loss=loss, optimizer=optim, metrics=metrics)

    print(model.summary())
    
    # save model into ong
    if save_model_png != False:
        save_model(model, output_path="models/summaries/resnet50.png")
    
    return model
