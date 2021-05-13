import gin
import tensorflow as tf
from keras import Input
from keras.layers import Dense, Conv1D, Conv2D
from keras.activations import relu
from keras.regularizers import l2, l1_l2
from keras.optimizers import Adam
from keras.models import Model
from networks.network import Network


@gin.configurable
class EncodingNetwork(Network):

    def __init__(self, state_shape,
                 conv_layer_params=None, fc_layer_params=None, activation=relu, dtype=tf.float32,
                 name='EncodingNetwork', conv_type='2d'):
        super().__init__(name=name)

        kernel_initializer = tf.keras.initializers.VarianceScaling(
            scale=2.0, mode='fan_in', distribution='truncated_normal')

        layers = []

        if conv_layer_params:
            if conv_type == '2d':
                conv_layer_type = tf.keras.layers.Conv2D
            elif conv_type == '1d':
                conv_layer_type = tf.keras.layers.Conv1D
            else:
                raise ValueError('unsupported conv type of %s. Use 1d or 2d' % (
                    conv_type))

            for config in conv_layer_params:
                if len(config) == 4:
                    (filters, kernel_size, strides, dilation_rate) = config
                elif len(config) == 3:
                    (filters, kernel_size, strides) = config
                    dilation_rate = (1, 1) if conv_type == '2d' else (1,)
                else:
                    raise ValueError(
                        'only 3 or 4 elements permitted in conv_layer_params tuples')
                layers.append(
                    conv_layer_type(
                        filters=filters,
                        kernel_size=kernel_size,
                        strides=strides,
                        dilation_rate=dilation_rate,
                        activation=activation,
                        kernel_initializer=kernel_initializer,
                        dtype=dtype))

        layers.append(tf.keras.layers.Flatten())

        if fc_layer_params:
            for num_units in fc_layer_params:
                kernal_regularizer = None # if necessary sholud have wheight decay param as in tf
                layers.append(
                    tf.keras.layers.Dense(
                        num_units,
                        activation=activation,
                        kernel_initializer=kernel_initializer,
                        kernel_regularizer=kernal_regularizer,
                        dtype=dtype))

        # Pull out the nest structure of the preprocessing layers. This avoids
        # saving the original kwarg layers as a class attribute which Keras would
        # then track.
        self._postprocessing_layers = layers
        self.built = True  # Allow access to self.variables

    def call(self, inputs, training=False, mask=None):
        states = inputs
        for layer in self._postprocessing_layers:
            states = layer(states, training=training)
        return states

    def get_config(self):
        pass
