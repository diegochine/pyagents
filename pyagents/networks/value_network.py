import gin
import tensorflow as tf
from pyagents.networks.network import Network
from pyagents.networks.encoding_network import EncodingNetwork


@gin.configurable
class ValueNetwork(Network):

    def __init__(self,
                 state_shape,
                 preprocessing_layers=None,
                 conv_layer_params=None,
                 fc_layer_params=(64, 64),
                 dropout_params=None,
                 activation='relu',
                 name='ValueNetwork',
                 trainable=True,
                 dtype=tf.float32):
        super(ValueNetwork, self).__init__(name, trainable, dtype)
        self._config = {'state_shape': state_shape,
                        'preprocessing_layers': [lay.config() for lay in preprocessing_layers]
                        if preprocessing_layers else [],
                        'conv_layer_params': conv_layer_params if conv_layer_params else [],
                        'fc_layer_params': fc_layer_params if fc_layer_params else [],
                        'dropout_params': dropout_params if dropout_params else [],
                        'activation': activation,
                        'name': name}
        if all(p is None for p in (preprocessing_layers, conv_layer_params, fc_layer_params)):
            self._encoder = None
        else:
            self._encoder = EncodingNetwork(
                state_shape,
                preprocessing_layers=preprocessing_layers,
                conv_layer_params=conv_layer_params,
                fc_layer_params=fc_layer_params,
                dropout_params=dropout_params,
                activation=activation,
            )
        self._value_head = tf.keras.layers.Dense(1)
        self.build((None, state_shape))

    def get_config(self):
        config = super().get_config()
        config.update(self._config)
        return config

    def call(self, inputs, training=True, mask=None):
        if self._encoder is not None:
            state = self._encoder(inputs, training=training)
        else:
            state = inputs
        value = self._value_head(state, training=training)
        return value
