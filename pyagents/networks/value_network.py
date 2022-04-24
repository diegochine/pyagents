import gin
import tensorflow as tf
from pyagents.networks.network import Network, NetworkOutput
from pyagents.networks.encoding_network import EncodingNetwork


@gin.configurable
class ValueNetwork(Network):

    def __init__(self,
                 state_shape,
                 conv_params=None,
                 fc_params=(64, 64),
                 dropout_params=None,
                 activation='relu',
                 name='ValueNetwork',
                 trainable=True,
                 dtype=tf.float32):
        super(ValueNetwork, self).__init__(name, trainable, dtype)
        self._config = {'state_shape': state_shape,
                        'conv_params': conv_params if conv_params else [],
                        'fc_params': fc_params if fc_params else [],
                        'dropout_params': dropout_params if dropout_params else [],
                        'activation': activation,
                        'name': name}
        if conv_params is None and fc_params is None:
            self._encoder = None
        else:
            self._encoder = EncodingNetwork(
                state_shape,
                conv_params=conv_params,
                fc_params=fc_params,
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
        return NetworkOutput(critic_values=value)

