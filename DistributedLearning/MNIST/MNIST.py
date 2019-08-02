"""
Train Model on MNIST using TF2.0
"""

import tensorflow as tf

# Network Characteristics
C_LAYER_FILTERS = 32
C_LAYER_SIZE = 3
H_LAYER1_SIZE = 128
H_LAYER2_SIZE = 32
DROPOUT_RATE = 0.2

# Training parameters
BATCH_SIZE = 128
EPOCHS = 5
VAL_SPLIT = 0.2

# Load MNIST Dataset
(IMG_TRAIN, LAB_TRAIN), (IMG_TEST, LAB_TEST) = tf.keras.datasets.mnist.load_data()
IMG_TRAIN, IMG_TEST = IMG_TRAIN / 255.0, IMG_TEST / 255.0

IMG_TRAIN = IMG_TRAIN[..., tf.newaxis]
IMG_TEST = IMG_TEST[..., tf.newaxis]


class MNIST_Model(tf.keras.Model):
    def __init__(self, is_cnn=False, dropout=None):
        self.is_cnn = is_cnn

        super(MNIST_Model, self).__init__()

        if self.is_cnn:
            self.conv1 = tf.keras.layers.Conv2D(C_LAYER_FILTERS, C_LAYER_FILTERS, activation='relu')
        else:
            self.dense1 = tf.keras.layers.Dense(H_LAYER1_SIZE, 'relu')
        self.flat = tf.keras.layers.Flatten()
        self.dense2 = tf.keras.layers.Dense(H_LAYER2_SIZE, 'relu')
        if dropout is None:
            self.dropout = tf.keras.layers.Dropout(DROPOUT_RATE)
        else:
            self.dropout = dropout
        self.dense3 = tf.keras.layers.Dense(10, 'softmax')

    def call(self, x):
        if self.is_cnn:
            x = self.conv1(x)
            x = self.flat(x)
        else:
            x = self.flat(x)
            x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x


if __name__ == '__main__':
    mdl = MNIST_Model(False)
    mdl.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    mdl.fit(IMG_TRAIN, LAB_TRAIN,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_split=VAL_SPLIT)

    mdl.evaluate(IMG_TEST, LAB_TEST)

