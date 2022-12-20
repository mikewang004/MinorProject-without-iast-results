import tensorflow as tf
from tensorflow import keras

(x_train, y_train), (x_val,y_val) = keras.datasets.fashion_mnist.load_data()