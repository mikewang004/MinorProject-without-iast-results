import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses

new_model = tf.keras.models.load_model('saved_model/my_model')
new_model.summary()

#examples = ["The movie was great!","The movie was okay.","The movie was terrible..."]

#predictans = new_model.predict(examples)