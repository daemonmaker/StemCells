"""
Train Model on MNIST dataset using distributed method with global observer
"""

import numpy as np
import multiprocessing
import DistributedLearning.MNIST.MNIST as MNIST
import os

# Training Parameters
N_NODES = 1
UPDATE_RATE = 10
VAL_SPLIT = 0.2
BATCH_SIZE = 128
EPOCHS = 5
RESULTS_DIR = 'results/'
GLOBAL_RESULT_FILE = 'glob_res.h5'

# Separate Test and validation sets
perm = np.random.permutation(MNIST.IMG_TRAIN.shape[0])
val_size = int(MNIST.IMG_TRAIN.shape[0]*VAL_SPLIT)
IMG_VALID = MNIST.IMG_TRAIN[perm[:val_size]]
LAB_VALID = MNIST.LAB_TRAIN[perm[:val_size]]
IMG_TRAIN = MNIST.IMG_TRAIN[perm[val_size:]]
LAB_TRAIN = MNIST.LAB_TRAIN[perm[val_size:]]
IMG_EVAL = IMG_TRAIN[:BATCH_SIZE]
LAB_EVAL = LAB_TRAIN[:BATCH_SIZE]
N_BATCHES = int(np.ceil(IMG_TRAIN.shape[0]/BATCH_SIZE))


class Node(multiprocessing.Process):
    """
    Worker Class
    If global observer, collects weights periodically and averages them
    Otherwise, performs training
    """
    def __init__(self, conn, idx=-1, **kwargs):
        super(Node, self).__init__(**kwargs)

        if idx == -1:
            self.is_global = True
        else:
            self.is_global = False
            self.idx = idx

        self.conn = conn
        self.global_weights = None

    def run(self):
        mdl = MNIST.MNIST_Model()
        mdl.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        mdl.predict(MNIST.IMG_TRAIN[0:1])

        if self.is_global:
            for conn in self.conn:
                conn.send(mdl.get_weights())
            update_no = 0
        else:
            perm = np.random.permutation(IMG_TRAIN.shape[0])
            batch = 0
            mdl.set_weights(self.conn.recv())

        running = True
        epoch = 0

        while running:
            if self.is_global:
                # accumulate results from workers
                new_weights = [np.zeros_like(w) for w in mdl.get_weights()]
                metrics = [0.0 for _ in mdl.metrics_names]
                for conn in self.conn:
                    loc_weights, loc_metrics = conn.recv()
                    for i in range(len(new_weights)):
                        new_weights[i] += loc_weights[i]
                    for i in range(len(metrics)):
                        metrics[i] += loc_metrics[i]

                # update global weights
                for i in range(len(new_weights)):
                    new_weights[i] = new_weights[i]/N_NODES

                # send new weights to workers
                for conn in self.conn:
                    conn.send(new_weights)

                # report results of update
                update_no += 1

                out = 'update %d complete! Metrics: '%(update_no,)
                for lab, val in zip(mdl.metrics_names, metrics):
                    out += '(%s: %f)' % (lab, val/N_NODES)
                print(out)

                # if end of epoch, perform validation
                if update_no >= N_BATCHES/UPDATE_RATE:
                    epoch += 1
                    update_no = 0
                    mdl.set_weights(new_weights)

                    val_results = mdl.evaluate(IMG_VALID, LAB_VALID, verbose=0)
                    out = 'Epoch %d/%d Complete! Results: '%(epoch, EPOCHS)
                    for lab, val in zip(mdl.metrics_names, val_results):
                        out += '(%s: %f)'%(lab, val)
                    print(out)
            else:
                # perform training steps
                for i in range(UPDATE_RATE):
                    bat_img = IMG_TRAIN[perm[batch*BATCH_SIZE:min((batch+1)*BATCH_SIZE, IMG_TRAIN.shape[0])]]
                    bat_lab = LAB_TRAIN[perm[batch*BATCH_SIZE:min((batch+1)*BATCH_SIZE, LAB_TRAIN.shape[0])]]
                    batch += 1
                    mdl.train_on_batch(bat_img, bat_lab)
                    if batch >= N_BATCHES:
                        epoch += 1
                        batch = 0
                        perm = np.random.permutation(IMG_TRAIN.shape[0])
                        break

                # Synchronize with global observer
                weights = mdl.get_weights()
                metrics = mdl.evaluate(IMG_EVAL, LAB_EVAL, verbose=0)
                self.conn.send([weights, metrics])
                mdl.set_weights(self.conn.recv())

            if epoch >= EPOCHS:
                running = False

        if self.is_global:
            for conn in self.conn:
                conn.close()
            mdl.save_weights(RESULTS_DIR+GLOBAL_RESULT_FILE)
            print('Training Complete!  Results saved!')
        else:
            self.conn.close()


if __name__ == '__main__':
    # Make sure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Create Pipes
    conns = [multiprocessing.Pipe() for _ in range(N_NODES)]

    # Create nodes
    global_node = Node([conns[i][0] for i in range(N_NODES)], name='Node_g')
    nodes = [Node(conns[i][1], i, name='Node_%d'%i) for i in range(N_NODES)]

    # Start Processes
    global_node.start()
    for n in nodes:
        n.start()

    # Join Processes
    global_node.join()
    for n in nodes:
        n.join()

    # Evaluate Results
    mdl = MNIST.MNIST_Model()
    mdl.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    mdl.predict(MNIST.IMG_TRAIN[0:1])
    mdl.load_weights(RESULTS_DIR+GLOBAL_RESULT_FILE)
    mdl.evaluate(MNIST.IMG_TEST, MNIST.LAB_TEST)
