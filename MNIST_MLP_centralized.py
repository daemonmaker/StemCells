"""
MNIST_MLP_centralized.py

Train MLP classifier on MNIST dataset
"""

import tensorflow as tf
import numpy as np
import _pickle as pkl
from tqdm import trange
import os
import matplotlib.pyplot as plt

# Setup training
SESS = tf.Session()
TRAINING = tf.placeholder_with_default(False, (), 'training')

# Network Parameters
H_SIZE = 512
DROPOUT_RATE = 0.2

# Training Parameters
LR = 0.001
PATIENCE = 10
BAT_SIZE = 128
VAL_PERC = 0.2
N_EPOCH = 100

# Data Storage Information
HOME = 'data/'
NET_FILENAME = 'MNIST_MLP_cent.pkl'

# Load MNIST Dataset
MNIST = tf.keras.datasets.mnist

(IMG_TRAIN, LAB_TRAIN), (IMG_TEST, LAB_TEST) = MNIST.load_data()
IMG_TRAIN = IMG_TRAIN/255.0
IMG_TEST = IMG_TEST/255.0
LAB_TRAIN_1H = np.zeros((LAB_TRAIN.shape[0], 10))
LAB_TEST_1H = np.zeros((LAB_TEST.shape[0], 10))

for i in range(LAB_TRAIN.shape[0]):
    LAB_TRAIN_1H[i, LAB_TRAIN[i]] = 1
for i in range(LAB_TEST.shape[0]):
    LAB_TEST_1H[i, LAB_TEST[i]] = 1

LAB_TRAIN = LAB_TRAIN_1H
LAB_TEST = LAB_TEST_1H

TRAIN_SIZE = IMG_TRAIN.shape[0]
VAL_SIZE = int(TRAIN_SIZE*VAL_PERC)
TRAIN_SIZE = TRAIN_SIZE - VAL_SIZE
TEST_SIZE = IMG_TEST.shape[0]
N_BATCH = int(np.ceil(TRAIN_SIZE/BAT_SIZE))

IMG_VALID = IMG_TRAIN[:VAL_SIZE]
IMG_TRAIN = IMG_TRAIN[VAL_SIZE:]

LAB_VALID = LAB_TRAIN[:VAL_SIZE]
LAB_TRAIN = LAB_TRAIN[VAL_SIZE:]

class MNIST_Class(object):
    def __init__(self, dropout=tf.keras.layers.Dropout(DROPOUT_RATE)):
        self.h_layer = tf.keras.layers.Dense(
                H_SIZE,
                tf.nn.leaky_relu,
                name='h')
        self.o_layer = tf.keras.layers.Dense(
                10,
                tf.nn.softmax,
                name='o')
        self.dropout = dropout
        self.scope = None

    def __call__(self, img):
        with tf.variable_scope('MNIST_Class') as scope:
            if self.scope is None:
                self.scope = scope

            img_flat = tf.keras.layers.Flatten()(img)
            h = self.h_layer(img_flat)
            if self.dropout is not None:
                h = self.dropout(h, training=TRAINING)
            o = self.o_layer(h)

            return o

    def save(self, filename):
        trainable_vars = self.h_layer.trainable_variables() + self.o_layer.trainable_variables()
        trainable_vals = SESS.run(trainable_vars)
        pkl.dump({var.name:val for var,val in zip(trainable_vars, trainable_vals)}, open(HOME+filename, 'wb'))

    def load(self, filename):
        trainable_vars = self.h_layer.trainable_variables() + self.o_layer.trainable_variables()
        load_data = pkl.load(open(HOME+filename, 'rb'))
        assign_ops = [var.assign(load_data[var.name]) for var in trainable_vars]
        SESS.run(assign_ops)

class Train_Centralized(object):
    def __init__(self):
        self.classifier = MNIST_Class()

        self.build()

    def build(self):
        self.img = tf.placeholder('float', [None, IMG_TRAIN.shape[1], IMG_TRAIN.shape[2]], 'img')
        self.label_tru = tf.placeholder('float', [None, 10], 'label_tru')
        self.label_est =self.classifier(self.img)
        self.loss = tf.reduce_mean(tf.squared_difference(self.label_tru, self.label_est))
        self.trainer = tf.train.AdamOptimizer(learning_rate=LR)
        self.train_op = self.trainer.minimize(self.loss)

    def train(self):
        hist = {}

        SESS.run(tf.global_variables_initializer())

        hist['loss_ave'] = np.zeros(N_EPOCH)
        hist['loss_val'] = np.zeros(N_EPOCH)

        min_weights=[]
        min_loss = np.inf
        min_loss_epoch = -1
        min_count = 0
        weights = tf.trainable_variables()

        for epoch in range(N_EPOCH):
            print('Epoch: %d/%d'%(epoch+1, N_EPOCH))
            perm = np.random.permutation(TRAIN_SIZE)
            batches = trange(N_BATCH)
            batches.set_description('Ave. Loss: N/A')

            for batch in batches:
                bat_img = IMG_TRAIN[perm[batch*BAT_SIZE:min((batch+1)*BAT_SIZE, TRAIN_SIZE)], :, :]
                bat_lab = LAB_TRAIN[perm[batch*BAT_SIZE:min((batch+1)*BAT_SIZE, TRAIN_SIZE)], :]

                _, loss = SESS.run([self.train_op, self.loss], feed_dict={self.img:bat_img, self.label_tru:bat_lab, TRAINING:True})

                hist['loss_ave'][epoch] += loss

                batches.set_description('Ave. Loss: %f'%(hist['loss_ave'][epoch]/(batch+1)))

            hist['loss_val'][epoch] = SESS.run(self.loss, feed_dict={self.img:IMG_VALID, self.label_tru:LAB_VALID, TRAINING:False})
            hist['loss_ave'][epoch] /= N_BATCH

            print('Epoch %d done.  Ave. Loss: %f, Val. Loss: %f'%(epoch, hist['loss_ave'][epoch], hist['loss_val'][epoch]))

            if hist['loss_val'][epoch] < min_loss:
                min_loss = hist['loss_val'][epoch]
                min_loss_epoch = epoch
                min_count = 0
                min_weights = SESS.run(weights)
            else:
                min_count += 1

            if min_count == PATIENCE:
                print('Training done, no improvment')
                hist['loss_ave'] = hist['loss_ave'][:epoch+1]
                hist['loss_val'] = hist['loss_val'][:epoch+1]
                break;
        SESS.run([var.assign(val) for var,val in zip(weights, min_weights)])
        hist['min_loss_epoch'] = min_loss_epoch

        test_lab = SESS.run(self.label_est, feed_dict={self.img:IMG_TEST, TRAINING:False})
        lab_num = np.argmax(test_lab, axis=1)
        tru_num = np.argmax(LAB_TEST, axis=1)
        hist['acc_rate'] = np.sum(lab_num==tru_num)/TEST_SIZE*100

        print('Accuracy: %f'%(hist['acc_rate']))

        return hist

if __name__=='__main__':
    os.makedirs(HOME, exist_ok=True)

    print('Setting up networks...')
    learner = Train_Centralized()

    print('Training...')
    hist = learner.train()

    print('Plotting results...')

    plt.figure()
    plt.plot(hist['loss_ave'])
    plt.plot(hist['loss_val'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training Loss', 'Valication Loss'])

    plt.show
