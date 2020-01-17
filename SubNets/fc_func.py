import os.path as op
import numpy as np

import keras
import keras.backend as KK
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist

model_file = 'fc_dense_model.json'
#weights_file = 'fc_dense_weights.h5'
weights_file = 'fc_dense_weights.npz'

load_model = True and op.exists(weights_file)
dropout = False
test_time_dropout = True

(x_train, y_train), (x_test, y_test) = mnist.load_data()
data, labels = x_train, keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

if load_model:
    model_weights = np.load(weights_file)['arr_0']

layer_idx = 0

dropout_inputs = []

# This returns a tensor
#inputs = Input(shape=(784,))
inputs = Input(shape=(28,28))
layer_idx += 1

flat_inputs = keras.layers.Flatten()(inputs)
layer_idx += 1

# a layer instance is callable on a tensor, and returns a tensor
layer = Dense(64, activation='relu')
x = layer(flat_inputs)
if load_model:
    layer.set_weights(model_weights[layer_idx])
layer_idx += 1

layer = keras.layers.BatchNormalization()
x = layer(x)
if load_model:
    layer.set_weights(model_weights[layer_idx])
layer_idx += 1

if dropout:
    x = keras.layers.Dropout(rate=0.2)(x)
    layer_idx += 1

def apply_test_time_dropout(args):
    x, dropout_mask = args
    return x*dropout_mask

if test_time_dropout:
    dropout_input = keras.layers.Input(batch_shape=(None, KK.int_shape(x)[-1]))
    x = keras.layers.Lambda(apply_test_time_dropout)([x, dropout_input])
    dropout_inputs.append(dropout_input)
    

layer = Dense(64, activation='relu')
x = layer(x)
if load_model:
    layer.set_weights(model_weights[layer_idx])
layer_idx += 1

layer = keras.layers.BatchNormalization()
x = layer(x)
if load_model:
    layer.set_weights(model_weights[layer_idx])
layer_idx += 1


if dropout:
    x = keras.layers.Dropout(rate=0.2)(x)
    layer_idx += 1

if test_time_dropout:
    dropout_input = keras.layers.Input(batch_shape=(None, KK.int_shape(x)[-1]))
    x = keras.layers.Lambda(apply_test_time_dropout)([x, dropout_input])
    dropout_inputs.append(dropout_input)

layer = Dense(10, activation='softmax')
predictions = layer(x)
if load_model:
    layer.set_weights(model_weights[layer_idx])

# This creates a model that includes
# the Input layer and three Dense layers
model_inputs = [inputs]
if test_time_dropout:
    model_inputs = model_inputs + dropout_inputs
model_outputs = [predictions]
model = Model(inputs=model_inputs, outputs=model_outputs)

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

if not load_model:
    model.fit(data, labels, epochs=1, validation_data=(x_test, y_test))  # starts training

    model_json = model.to_json()
    with open(model_file, 'w') as json_file:
        json_file.write(model_json)

    model_weights = []
    for layer in model.layers:
        model_weights.append(layer.get_weights())
    np.savez(weights_file, model_weights)


score = model.evaluate([x_test, np.ones((10000, 64)), np.ones((10000, 64))], [y_test])
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])

test_time_dropout_rate = 0.2
if test_time_dropout:
    mask_count = 100
    all_dropout_mask_values = []
    for mask_idx in range(mask_count):
        dropout_mask_values = []
        for idx, dropout_input in enumerate(dropout_inputs):
            dropout_mask_value = (np.random.rand(1, KK.int_shape(dropout_input)[-1]) > test_time_dropout_rate).astype(np.int8)
            dropout_mask_values.append(dropout_mask_value)
        all_dropout_mask_values.append(dropout_mask_values)

    scores = []
    test = 0
    for run_idx, dropout_mask_values in enumerate(all_dropout_mask_values):
        tiled_dropout_mask_values = [np.tile(dropout_mask_value, (y_test.shape[0], 1)) for dropout_mask_value in dropout_mask_values]
        score = model.evaluate([x_test,] + tiled_dropout_mask_values, [y_test])
        if run_idx == 0:
            scores = score
        else:
            scores[0] += score[0]
            scores[1] += score[1]
            print('Test loss: ', score[0])
            print('Test accuracy: ', score[1])
        test += 1

    print('Test loss: ', scores[0]/mask_count)
    print('Test accuracy: ', scores[1]/mask_count)
