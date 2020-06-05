import tensorflow as tf

class CustomDropout(tf.keras.layers.Layer):
    def __init__(self):
        super(CustomDropout, self).__init__()

    def call(self, input):
        x, mask = input
        return tf.multiply(x, mask)


class FullyConnectedModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super(FullyConnectedModel, self).__init__()
        self.kwargs = kwargs

        self.hidden_layer_size = self.kwargs['hidden_layer_size']
        self.dropout_rate = self.kwargs['dropout_rate']
        self.batch_size = self.kwargs['batch_size']    
        self.epochs = self.kwargs['epochs']
        self.output_size = self.kwargs['output_size']

        #self.img_input = tf.keras.Input(batch_shape=(None, 28, 28), name='img_input')
        #self.mask_input = tf.keras.Input(batch_shape=(None, self.hidden_layer_size), name='mask_input')

        self.layer1 = tf.keras.layers.Flatten(input_shape=(None, 28, 28), name='flatten')
        self.layer2 = tf.keras.layers.Dense(self.hidden_layer_size, activation='relu', name='encoder')
        self.layer3 = CustomDropout()
        self.layer4 = tf.keras.layers.Dense(self.output_size, name='output')

    def call(self, inputs):
        h = self.layer1(inputs[0])
        h = self.layer2(h)
        h = self.layer3([h, inputs[1]])
        self.prediction = self.layer4(h)

        return self.prediction


if __name__ == "__main__":
    pass