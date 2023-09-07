import tensorflow as tf
from tensorflow.python.keras import layers, models
import keras_cv
import matplotlib.pyplot as plt

from pathlib import Path
import tensorflow_datasets as tfds

import fracatlas

# 80% train, 20% test
train_ds, test_ds = tfds.load("fracatlas", split=["train[:80%]", "test[80%:]"])
