import tensorflow as tf
import numpy as np
import random
import string

class CustomDropout(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CustomDropout, self).__init__(**kwargs)

    def call(self, input):
        x, mask = input
        return tf.multiply(x, mask)


class FullyConnectedModel(tf.keras.Model):
    def __init__(self, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = [random.choice(string.ascii_letters + string.digits) for _ in range(10)]
        super(FullyConnectedModel, self).__init__()
        self.kwargs = kwargs

        #self.name = self.kwargs.get('name', [random.choice(string.ascii_letters + string.digits) for _ in range(10)])
        self.hidden_layer_size = self.kwargs['hidden_layer_size']
        self.output_size = self.kwargs['output_size']

        self.inputs = {
            'img': tf.keras.Input(batch_shape=(None, 28, 28), name='img_input'),
            'mask': tf.keras.Input(batch_shape=(None, self.hidden_layer_size), name='mask_input'),
        }
      
        self.layer1 = tf.keras.layers.Flatten(input_shape=(None, 28, 28), name='{}_flatten'.format(self.name))

        self.layer2 = tf.keras.layers.Dense(self.hidden_layer_size, activation='relu', name='{}_encoder'.format(self.name))
        self.layer3 = CustomDropout(name='{}_CustomDropout'.format(self.name))

        self.layer4 = tf.keras.layers.Dense(self.output_size, name='{}_output'.format(self.name))

        self.accumulators = {}
 
    def init_accumulators(self):
        self.accumulators[self.layer2.name] = self._build_accumulator(self.layer2)
        self.accumulators[self.layer4.name] = self._build_accumulator(self.layer4)

    def _build_accumulator(self, layer):
        return [np.zeros(weights.shape) for weights in layer.trainable_weights]
    
    def acculumate_grads(self, grads):
        pass

    def copy_weights(self, source_model):
        '''
        for global_layer, self_layer in zip(model.layers, self.layers):
            print(global_layer.name)
            print(global_layer.get_weights())
            print(self_layer.name)
            print(self_layer.get_weights())
            for global_weights, self_weights in zip(global_layer.trainable_weights, self_layer.trainable_weights):
                print(global_weights)
                print(self_weights)
        '''
        
        '''
        print('#####################################')
        print("global weights: ", model.trainable_weights)
        #print('testing: ', model.trainable_weights[0])
        print('testing: ', model.trainable_weights[0].read_value())
        print('-------------------------------------')
        print("self weights: ", self.trainable_weights)
        print('-------------------------------------')
        print('-------------------------------------')
        '''
        '''
        for source_weights, destination_weights in zip(model.trainable_weights, self.trainable_weights):
            destination_weights.assign(source_weights)
        '''
        #self.set_weights(source_model.get_weights)
        pass
        #exit()

    def call(self, inputs):
        h = self.layer1(inputs[0])
        h = self.layer2(h)
        h = self.layer3([h, inputs[1]])
        self.prediction = self.layer4(h)

        return self.prediction


if __name__ == "__main__":
    pass