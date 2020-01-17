from __future__ import print_function
import keras
from keras.datasets import mnist, fashion_mnist, cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, ReLU
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions

dataset = 'cifar10'
# the data, split between train and test sets
if dataset == 'cifar10':
    img_rows, img_cols, img_chans = 32, 32, 3
    flatten_dim = 512
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    input_shape = (img_rows, img_cols, img_chans)
else:
    img_rows, img_cols, img_chans = 28, 28, 1
    flatten_dim = 128
    #(x_train, y_train), (x_test, y_test) = mnist.load_data()
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (img_chans, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, img_chans)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), input_shape=input_shape))
model.add(BatchNormalization())
model.add(ReLU())
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization())
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(ReLU())
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(flatten_dim))
model.add(Dropout(0.5))
model.add(ReLU())
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

print(model.summary())

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])