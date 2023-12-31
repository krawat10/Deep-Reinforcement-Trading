import tensorflow as tf


class DropoutNoisy(tf.keras.layers.Layer):
    def __init__(self, rate, std, activation, kernel_regularizer, name, **kwargs):
        super(DropoutNoisy, self).__init__(**kwargs)
        self.rate = rate
        self.std = std
        self.activation = activation
        self.kernel_regularizer = kernel_regularizer
        self.name = name

    def call(self, inputs):
        # Apply dropout to the input tensor
        x = tf.nn.dropout(inputs, rate=self.rate, name=self.name + '_dropout')

        # Add random noise to the tensor with specified standard deviation
        x += tf.random.normal(shape=tf.shape(x), stddev=self.std, name=self.name + '_noise')

        # Apply the specified activation function to the tensor
        if self.activation == 'relu':
            x = tf.nn.relu(x, name=self.name + '_relu')
        elif self.activation == 'sigmoid':
            x = tf.nn.sigmoid(x, name=self.name + '_sigmoid')
        # Add more elif statements here for additional activation functions

        # Apply the specified regularization to the tensor
        if self.kernel_regularizer is not None:
            x = self.kernel_regularizer(x)

        return x
