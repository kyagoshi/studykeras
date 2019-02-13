from keras.datasets import boston_housing
import numpy as np
from keras import models
from keras import layers

(train_data, train_datasets), (test_data, test_targets) = boston_housing.load_data()

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

def bulid_model():
    