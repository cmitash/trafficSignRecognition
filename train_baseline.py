from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import SGD
import numpy 
import h5py

NUM_CLASSES = 43
IMG_SIZE = 48

def lr_schedule(epoch):
    return lr*(0.1**int(epoch/10))

def cnn_model():
    model = Sequential()

    model.add(Convolution2D(32, (3, 3), padding='same', input_shape=(IMG_SIZE, IMG_SIZE,3), activation='relu'))
    model.add(Convolution2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.2))

    model.add(Convolution2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    return model


model = cnn_model()
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
                    ModelCheckpoint('data/model/model_baseline.h5',save_best_only=True)]
         )