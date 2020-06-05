import threading
import tensorflow as tf
import numpy as np
from custom_layers import *


class Worker(threading.Thread):
    def __init__(self, group=None, target=None, name=None,
                     args=(), kwargs=None, *, daemon=None):
        super().__init__(group=group, target=target, name=name,
                         daemon=daemon)
        self.idx = args
        self.kwargs = kwargs
        
        # Create the model
        self.coordinator = self.kwargs['coordinator']
        self.hidden_layer_size = self.kwargs['hidden_layer_size']
        self.dropout_rate = self.kwargs['dropout_rate']
        self.batch_size = self.kwargs['batch_size']    
        self.epochs = self.kwargs['epochs']

        self.img_input = tf.keras.Input(shape=(28, 28), name='img_input')
        self.mask_input = tf.keras.Input(shape=(self.hidden_layer_size), name='mask_input')

        layer1 = tf.keras.layers.Flatten(name='flatten')
        print(layer1)
        layer2 = tf.keras.layers.Dense(self.hidden_layer_size, activation='relu', name='encoder')
        print(layer2)
        layer3 = CustomDropout()
        print(layer3)
        layer4 = tf.keras.layers.Dense(10, name='output')
        print(layer4)

        h = layer1(self.img_input)
        print("h: ", h)
        h = layer2(h)
        print("h: ", h)
        h = layer3([h, self.mask_input])
        print("h: ", h)
        self.output = layer4(h)
        print("output: ", self.output)

        self.model = tf.keras.Model(inputs=[self.img_input, self.mask_input], outputs=self.output)
        self.model.summary()
        tf.keras.utils.plot_model(self.model, show_shapes=True)
        
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        
        self.model.compile(optimizer='sgd', loss=self.loss_fn, metrics=['accuracy'])

    def create_masks(self, batch_size):
        mask = np.random.rand(1, self.hidden_layer_size) <= self.dropout_rate
        mask = mask.astype(np.int32)
        mask = np.tile(mask, (batch_size, 1))
        return mask

    def run(self):
        with self.coordinator.stop_on_exception():
            mnist = tf.keras.datasets.fashion_mnist

            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            x_train, x_test = x_train / 255.0, x_test / 255.0

            train_samples = x_train.shape[0]
            batches = train_samples // self.batch_size
            dropout_rate = 1-0.2

            copies = int(train_samples/self.batch_size)
            extra = train_samples - copies*self.batch_size

            # Train the model
            for epoch_idx in range(self.epochs):
                loss_values = []
                for batch_idx in range(batches):
                    mask = self.create_masks(self.batch_size)
                    current_batch = x_train[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]
                    labels = y_train[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]
                    loss_value = self.model.train_on_batch([current_batch, mask], y=labels)
                    loss_values.append(loss_value)

                excess_samples = train_samples % self.batch_size
                if excess_samples > 0:
                    mask = self.create_masks(excess_samples)
                    current_batch = x_train[batches*self.batch_size:]
                    labels = y_train[batches*self.batch_size:]
                    loss_value = self.model.train_on_batch([current_batch, mask], y=labels)
                    loss_values.append(loss_value)

                average_loss = np.sum(loss_values) / train_samples
                print("Thread: ", self.idx, "\tEpoch: ", epoch_idx, "\tAverage loss: ", average_loss, "\tAccuracy: ", 1-average_loss)
