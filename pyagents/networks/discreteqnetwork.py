import gin
import tensorflow as tf

from pyagents.layers.qlayer import QLayer
from pyagents.networks.network import Network
from pyagents.networks.encoding_network import EncodingNetwork


@gin.configurable
class DiscreteQNetwork(Network):
    """A DQN for discrete action spaces, taking in input a state and outputting [Q(s, a_1), ..., Q(s, a_n)]"""

    def __init__(self, state_shape, action_shape,
                 conv_params=None, fc_params=(64, 64), dropout_params=None, activation='relu',
                 name='QNetwork', trainable=True, dtype=tf.float32, dueling=True):
        super().__init__(name=name, trainable=trainable, dtype=dtype)
        self._config = {'state_shape': state_shape,
                        'action_shape': action_shape,
                        'conv_params': conv_params if conv_params else tuple(),
                        'fc_params': fc_params if fc_params else tuple(),
                        'dropout_params': dropout_params if dropout_params else tuple(),
                        'activation': activation,
                        'dueling': dueling,
                        'name': name}
        self._encoder = EncodingNetwork(
            state_shape,
            conv_params=conv_params,
            fc_params=fc_params[:-1],
            dropout_params=dropout_params,
            activation=activation,
            name=name
        )
        self._q_layer = QLayer(action_shape, units=fc_params[-1], dropout=dropout_params, dueling=dueling)

    def call(self, inputs, training=False, mask=None):
        state = self._encoder(inputs, training=training)
        q_values = self._q_layer(state, training=training)
        return q_values

    def get_config(self):
        config = super().get_config()
        config.update(self._config)
        return config
