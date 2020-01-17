import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, ReLU
from tensorflow.keras.datasets import fashion_mnist, mnist
from tensorflow.keras import backend as KK
from tensorflow.keras.optimizers import Adam, RMSprop


class FullyConnectedClassifier(tf.keras.Model):
    def __init__(self, num_classes, dropout=True):
        super(FullyConnectedClassifier, self).__init__()

        self.dn_1 = Dense(1000, name='dense_1')
        self.bn_1 = BatchNormalization()
        self.relu_1 = ReLU()

        if dropout is not None:
            self.dropout_1 = Masker(0.2, name='dropout_1', state='on')
        else:
            self.dropout_1 = None

        self.dn_2 = Dense(10000, name='dense_2')
        self.bn_2 = BatchNormalization()
        self.relu_2 = ReLU()

        if dropout is not None:
            self.dropout_2 = Masker(0.5, name='dropout_2', state='off')
        else:
            self.dropout_2 = None

        self.prediction = Dense(num_classes, activation=tf.nn.softmax, name='output')

    def call(self, inputs):
        hidden = inputs
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


class MaskApply(tf.keras.callbacks.Callback):
    def __init__(self, masker, name=None):
        super(MaskApply, self).__init__()

        self.name = name
        self.masker = masker
        self.count = 0

    #def on_train_batch_begin(self, batch, logs=None):
    def on_epoch_begin(self, epoch, logs=None):
        if self.masker.mask is None:
            return
        '''
        mask = KK.get_value(self.masker.mask)
        print("mask: ", mask)
        mask = np.zeros((1, shape))
        '''

        if self.count > 2:
            shape = KK.int_shape(self.masker.mask)[1]

            mask = np.random.random((1, shape))
            mask[mask < self.masker.rate] = 0.
            mask[mask >= self.masker.rate] = 1.

            KK.set_value(self.masker.mask, mask)

        self.count += 1


class Masker(tf.keras.layers.Layer):#, tf.keras.callbacks.Callback):
    def __init__(self, rate, name=None, state='random'):
        super(Masker, self).__init__(name=name)

        self.count = 0
        self.rate = rate
        self.mask = None
        self.trainable = False
        self.state = state

    def call(self, inputs):
        if self.mask is None:
            shape = tf.keras.backend.int_shape(inputs)[1]
            mask = np.random.random((1, shape))
            if self.state == 'on':
                mask[mask <= self.rate] = 1.
                mask[mask >= self.rate] = 1.
            elif self.state == 'off':
                mask[mask <= self.rate] = 0.5
                mask[mask >= self.rate] = 0.5
            else:  # random
                mask[mask <= self.rate] = 0.
                mask[mask >= self.rate] = 1.
            self.mask = tf.keras.backend.variable(mask)
        return tf.math.multiply(inputs, self.mask)


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    flattened_dim = x_train.shape[1]*x_train.shape[2]
    x_train = (x_train.astype(np.float32) / 255.0).reshape(x_train.shape[0], flattened_dim)
    x_test = (x_test.astype(np.float32) / 255.0).reshape(x_test.shape[0],  flattened_dim)

    apply_masks = True
    fc = FullyConnectedClassifier(10, dropout=apply_masks)
    if apply_masks:
        masked_1 = MaskApply(fc.dropout_1)
        masked_2 = MaskApply(fc.dropout_2)
        callbacks = [masked_1, masked_2]
    else:
        dropout = None
        callbacks = None
    optimizer = Adam(learning_rate=0.0001)
    #optimizer = RMSprop()
    fc.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    fc.fit(x_train, y_train, batch_size=96, epochs=100, validation_split=0.2, callbacks=callbacks)

    fc.evaluate(x_test, y_test)

