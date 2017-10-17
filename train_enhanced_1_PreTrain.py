from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import SGD
from keras.models import load_model
import numpy 
import h5py

NUM_CLASSES = 43
IMG_SIZE = 48

def lr_schedule(epoch):
    return lr*(0.1**int(epoch/10))

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

    model.load_weights('data/pretrain_models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', by_name=True)

    return model

model = vgg16_model()

X = numpy.load('X.npy')
Y = numpy.load('Y.npy')

print ('X shape: ', X.shape)
print ('Y shape: ', Y.shape)

# let's train the model using SGD + momentum
lr = 0.01
batch_size = 32
nb_epoch = 30

sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
          optimizer=sgd,
          metrics=['accuracy'])

model.fit(X, Y,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_split=0.2,
          callbacks=[LearningRateScheduler(lr_schedule),
                    ModelCheckpoint('data/models/model_vgg16_pretrained_notop.h5',save_best_only=True)]
         )