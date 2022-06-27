import numpy as np
import tensorflow as tf


class NoisyLayer(tf.keras.layers.Layer):
    """https://arxiv.org/pdf/1706.10295.pdf"""
    def __init__(self, units,
                 activation='tanh',
                 kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)),
                 kernel_regularizer=None,
                 name='noisy',
                 dtype=tf.float32,
                 **kwargs):
        super(NoisyLayer, self).__init__(name=name, dtype=dtype, **kwargs)
        self.units = units
        self.activation = activation
        self._activation = tf.keras.activations.get(activation)
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer
        self.built = False

    def build(self, input_shape):
        in_dim = input_shape[-1]
        b_range = 1 / in_dim
        self.sigma_w = self.add_weight(
            shape=(in_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True
        )
        self.mu_w = self.add_weight(
            shape=(in_dim, self.units,),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True
        )
        self.sigma_b = self.add_weight(
            shape=(self.units,),
            initializer=tf.keras.initializers.RandomUniform(-b_range, b_range),
            trainable=True
        )
        self.mu_b = self.add_weight(
            shape=(self.units,),
            initializer=tf.keras.initializers.RandomUniform(-b_range, b_range),
            trainable=True
        )
        self.reset_noise()

    def reset_noise(self, stddev=0.5):
        def scale(x):
            return tf.sign(x) * tf.sqrt(tf.abs(x))
        eps_in = tf.random.normal((self.sigma_w.shape[0],), stddev=stddev)
        eps_out = tf.random.normal((self.units,), stddev=stddev)
        eps_in = scale(eps_in)
        eps_out = scale(eps_out)
        self.eps_w = tf.tensordot(eps_in, eps_out, axes=0)  # outer product to get (in_shape, units) matrix
        self.eps_b = eps_out

    def call(self, x, training=True):
        rank = x.shape.rank
        w = self.mu_w + self.sigma_w * self.eps_w
        b = self.mu_b + self.sigma_b * self.eps_b
        # dot product along last dimension of input vector
        x = tf.tensordot(x, w, axes=[[rank - 1], [0]]) + b
        if self._activation is not None:
            x = self._activation(x)
        return x

    def get_config(self):
        config = super(NoisyLayer, self).get_config()
        config.update({
            'units': self.units,
            'activation': self.activation,
            'use_bias': self.use_bias})
        return config
