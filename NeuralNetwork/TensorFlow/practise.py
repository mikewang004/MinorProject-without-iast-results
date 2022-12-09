import tensorflow as tf
from tensorflow import keras
print("TensorFlow version:",tf.__version__)
import numpy as np
from matplotlib import image
import cv2 as cv
import os

#Functions
(x_train, y_train), (x_val,y_val) = keras.datasets.fashion_mnist.load_data()
def preprocess(x,y):
    x = tf.cast(x,tf.float32)/255.0
    y = tf.cast(y,tf.int64)
    
    return x,y

def create_datasets(xs,ys,n_classes = 10):
    ys = tf.one_hot(ys,depth=n_classes)
    return tf.data.Dataset.from_tensor_slices((xs,ys))\
        .map(preprocess)\
        .shuffle(len(ys))\
        .batch(128)





train_dataset = create_datasets(x_train, y_train)
val_dataset = create_datasets(x_val, y_val)
'''
model = keras.Sequential([
    keras.layers.Reshape(target_shape=(28 * 28,), input_shape=(28, 28)),
    keras.layers.Dense(units=256, activation='relu'),
    keras.layers.Dense(units=192, activation='relu'),
    keras.layers.Dense(units=128, activation='relu'),
    keras.layers.Dense(units=10, activation='softmax')
])

model.compile(optimizer='adam', 
              loss=tf.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(
    train_dataset.repeat(),
    epochs = 50,
    steps_per_epoch = 500,
    validation_data = val_dataset.repeat(),
    validation_steps = 2
)


model.save('saved_model/')
| Label | Description |
|-------|-------------|
| 0     | T-shirt/top |
| 1     | Trouser     |
| 2     | Pullover    |
| 3     | Dress       |
| 4     | Coat        |
| 5     | Sandal      |
| 6     | Shirt       |
| 7     | Sneaker     |
| 8     | Bag         |
| 9     | Ankle boot  |

'''
model = keras.models.load_model('saved_model/')
predictions = model.predict(val_dataset)

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder,filename))
        gray_img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        gr_img = cv.resize(gray_img,(28,28))
        if img is not None:
            images.append(gr_img)
    return images

input_lst = load_images_from_folder("Images")
#cv.imwrite("Images/test_gray_resized.jpg",gr_img)


own_data = create_datasets(input_lst, [0,4,0,7])
own_pred = model.predict(own_data)
