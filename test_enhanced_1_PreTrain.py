import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
import numpy 
import h5py
import os
from skimage import color, exposure, transform
from skimage import io

NUM_CLASSES = 43
IMG_SIZE = 48

def preprocess_img(img):
    # Histogram normalization in v channel
    hsv = color.rgb2hsv(img)
    hsv[:,:,2] = exposure.equalize_hist(hsv[:,:,2])
    img = color.hsv2rgb(hsv)

    # central square crop
    min_side = min(img.shape[:-1])
    centre = img.shape[0]//2, img.shape[1]//2
    img = img[centre[0]-min_side//2:centre[0]+min_side//2,
              centre[1]-min_side//2:centre[1]+min_side//2,
              :]

    # rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))

    return img

test = pd.read_csv('data/GTSRB/GT-final_test.csv',sep=';')

def vgg16_model():
    model = Sequential()

    # Block 1
    model.add(Convolution2D(64, (3, 3), padding='same', input_shape=(IMG_SIZE, IMG_SIZE,3), activation='relu'))
    model.add(Convolution2D(64, (3, 3), padding='same', input_shape=(IMG_SIZE, IMG_SIZE,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),  strides=(2, 2)))

    # Block 2
    model.add(Convolution2D(128, (3, 3), padding='same', input_shape=(IMG_SIZE, IMG_SIZE,3), activation='relu'))
    model.add(Convolution2D(128, (3, 3), padding='same', input_shape=(IMG_SIZE, IMG_SIZE,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),  strides=(2, 2)))

    # Block 3
    model.add(Convolution2D(256, (3, 3), padding='same', input_shape=(IMG_SIZE, IMG_SIZE,3), activation='relu'))
    model.add(Convolution2D(256, (3, 3), padding='same', input_shape=(IMG_SIZE, IMG_SIZE,3), activation='relu'))
    model.add(Convolution2D(256, (3, 3), padding='same', input_shape=(IMG_SIZE, IMG_SIZE,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),  strides=(2, 2)))

    # Block 4
    model.add(Convolution2D(512, (3, 3), padding='same', input_shape=(IMG_SIZE, IMG_SIZE,3), activation='relu'))
    model.add(Convolution2D(512, (3, 3), padding='same', input_shape=(IMG_SIZE, IMG_SIZE,3), activation='relu'))
    model.add(Convolution2D(512, (3, 3), padding='same', input_shape=(IMG_SIZE, IMG_SIZE,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),  strides=(2, 2)))

    # Block 5
    model.add(Convolution2D(512, (3, 3), padding='same', input_shape=(IMG_SIZE, IMG_SIZE,3), activation='relu'))
    model.add(Convolution2D(512, (3, 3), padding='same', input_shape=(IMG_SIZE, IMG_SIZE,3), activation='relu'))
    model.add(Convolution2D(512, (3, 3), padding='same', input_shape=(IMG_SIZE, IMG_SIZE,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),  strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    return model

model = vgg16_model()
model.load_weights('data/models/model_vgg16_pretrained_notop.h5')

# Load test dataset
X_test = []
y_test = []
i = 0
for file_name, class_id  in zip(list(test['Filename']), list(test['ClassId'])):
    img_path = os.path.join('data/GTSRB/Final_Test/Images/',file_name)
    X_test.append(preprocess_img(io.imread(img_path)))
    y_test.append(class_id)
    
X_test = numpy.array(X_test)
y_test = numpy.array(y_test)

# predict and evaluate
y_pred = model.predict_classes(X_test)
print (y_pred)
acc = float(numpy.sum(y_pred==y_test))/numpy.size(y_pred)
print("Test accuracy = {}".format(acc))