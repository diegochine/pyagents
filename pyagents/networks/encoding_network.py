import gin
import tensorflow as tf
from pyagents.networks.network import Network


@gin.configurable
class EncodingNetwork(Network):

    def __init__(self,
                 state_shape,
                 conv_params=None,
                 fc_params=None,
                 dropout_params=None,
                 activation='relu',
                 dtype=tf.float32,
                 name='EncodingNetwork',
                 conv_type='2d'):
        super().__init__(name=name)
        self._state_shape = state_shape

        # TODO improve inizialization, allow initizializer to be passed as parameter
        kernel_initializer = tf.keras.initializers.GlorotNormal()

        layers = []

        if conv_params:
            if conv_type == '2d':
                conv_layer_type = tf.keras.layers.Conv2D
            elif conv_type == '1d':
                conv_layer_type = tf.keras.layers.Conv1D
            else:
                raise ValueError('unsupported conv type of %s. Use 1d or 2d' % (
                    conv_type))

            for config in conv_params:
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
        if dropout_params is None or isinstance(dropout_params, float):
            dropout_params = [dropout_params] * len(fc_params)
        else:
            assert isinstance(dropout_params, (tuple, list)), f'unrecognized type for dropout_params {type(dropout_params)} '
        if fc_params:
            assert len(fc_params) == len(dropout_params), f'params length do not match (fc: {len(fc_params)}, dropout: {len(dropout_params)})'
            for num_units, dropout in zip(fc_params, dropout_params):
                kernel_regularizer = None  # if necessary sholud have wheight decay param as in tf
                layers.append(
                    tf.keras.layers.Dense(
                        num_units,
                        activation=activation,
                        kernel_initializer=kernel_initializer,
                        kernel_regularizer=kernel_regularizer,
                        dtype=dtype))
                if dropout is not None:
                    layers.append(tf.keras.layers.Dropout(rate=dropout))

        # Pull out the nest structure of the preprocessing layers. This avoids
        # saving the original kwarg layers as a class attribute which Keras would
        # then track.
        self._postprocessing_layers = layers
        self.built = True  # Allow access to self.variables
        self._config = {'state_shape': state_shape,
                        'conv_layer_params': conv_params if conv_params else tuple(),
                        'fc_layer_params': fc_params if conv_params else tuple(),
                        'dropout_params': dropout_params if dropout_params else tuple(),
                        'activation': activation,
                        'dtype': dtype,
                        'name': name,
                        'conv_type': conv_type}

    def call(self, inputs, training=False, mask=None):
        states = inputs
        for layer in self._postprocessing_layers:
            states = layer(states, training=training)
        return states

    def get_config(self):
        config = super(EncodingNetwork, self).get_config()
        config.update(self._config)
        return config

    @classmethod
    def from_config(cls, config):
        """Be careful with the type of the preprocessing layers:
        they must be a list of already built layer and not a list
        of their configurations."""
        return cls(**config)
