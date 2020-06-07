import threading
import copy
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
        self.output_size = self.kwargs['output_size']
        self.learning_rate = self.kwargs['learning_rate']
        self.dropout_rate = self.kwargs['dropout_rate']
        self.batch_size = self.kwargs['batch_size']
        self.epochs = self.kwargs['epochs']
        self.global_model = self.kwargs['global_model']
        self.model = self.kwargs['model']
        self.update_batches = self.kwargs['update_batches']
        self.regenerate_masks = self.kwargs['regenerate_masks']
        self.mask_generation_batches = self.kwargs['mask_generation_batches']


        self.inputs = {
            'img': self.model.inputs['img'],
            'mask': self.model.inputs['mask'],
        }

        print(self.model.summary())

        #self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        #self.optimizer = tf.keras.optimizers.SGD(learning_rate=1e-2)
        self.loss_metric = tf.keras.metrics.Mean()
        self.accuracy = tf.keras.metrics.CategoricalAccuracy()


    def create_masks(self, batch_size):
        mask = np.random.rand(1, self.hidden_layer_size) <= self.dropout_rate
        mask = mask.astype(np.float32)
        print("np.sum(mask): ", np.sum(mask))
        print("mask: ", mask)
        mask = np.tile(mask, (batch_size, 1))
        return mask

    def run(self):
        with self.coordinator.stop_on_exception():
            (x_train, y_train), (x_valid, y_valid) = tf.keras.datasets.mnist.load_data()
            x_train = x_train.astype('float32') / 255
            x_valid = x_valid.astype('float32') / 255
            y_train = tf.keras.backend.one_hot(y_train, self.output_size)
            y_valid = tf.keras.backend.one_hot(y_valid, self.output_size)
            total_samples = x_train.shape[0]
            masks_train = self.create_masks(total_samples) # Creates the same mask for every sample, therefore it describes a single network
            masks_valid = np.ones((x_valid.shape))
            print("x_train.shape: ", x_train.shape)
            print("y_train.shape: ", y_train.shape)
            print("masks_train.shape: ", masks_train.shape)
            print("x_valid.shape: ", x_valid.shape)
            print("y_valid.shape: ", y_valid.shape)
            print("masks_valid.shape: ", masks_valid.shape)

            #train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
            train_dataset = tf.data.Dataset.from_tensor_slices((x_train, masks_train, y_train))
            train_dataset = train_dataset.shuffle(buffer_size=1024).batch(self.batch_size)
            print(train_dataset)
            '''
            valid_dataset = tf.data.Dataset.from_tensor_slices([x_valid, masks_valid, y_valid])
            valid_dataset = valid_dataset.batch(self.batch_size)
            print(valid_dataset)
            '''
            fc = self.model

            # Iterate over epochs.
            for epoch in range(self.epochs):
              print('Start of epoch %d' % (epoch,))
            
              #self.model.copy_weights(self.global_model)
              #self.loss_metric.reset_states()
              #self.accuracy.reset_states()
              if self.update_batches > 0:
                  fc.set_weights(copy.deepcopy(self.global_model.get_weights()))

              # Iterate over the batches of the dataset.
              #for step, (x_train, y_train) in enumerate(train_dataset):
              for step, (x_train, masks_train, y_train) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                  #masks = np.ones((x_train.shape[0], self.hidden_layer_size))
                  if self.regenerate_masks and (step == 0 or self.mask_generation_batches % step == 0):
                      masks_train = self.create_masks(x_train.shape[0])
                  #logits = fc([x_train, masks])
                  logits = fc([x_train, masks_train])
                  # Compute reconstruction loss
                  #loss = mse_loss_fn(y_train, logits)
                  loss = self.loss_fn(y_train, logits)

                self.accuracy.update_state(y_train, logits)

                grads = tape.gradient(loss, fc.trainable_weights)
                #print("grads: ", grads)
                self.optimizer.apply_gradients(zip(grads, fc.trainable_weights))
                self.optimizer.apply_gradients(zip(grads, self.global_model.trainable_weights))

                self.loss_metric(loss)

                if step % 100 == 0:
                  print('Epoch: %s\tThread: %s\tstep %s\tmean loss = %s\n\tmean accuracy = %s' % (epoch, str(self.idx), step, self.loss_metric.result(), self.accuracy.result()))

                if self.update_batches > 0 and step % self.update_batches == 0:
                    #print("Copying weights")
                    #self.loss_metric.reset_states()
                    #self.accuracy.reset_states()
                    fc.set_weights(copy.deepcopy(self.global_model.get_weights()))
