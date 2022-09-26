import gin
import numpy as np
import tensorflow as tf

from pyagents.layers.qlayer import QLayer
from pyagents.networks.network import Network, NetworkOutput
from pyagents.networks.encoding_network import EncodingNetwork


@gin.configurable
class QRQNetwork(Network):
    """A DQN for discrete action spaces, taking in input a state and outputting [Q(s, a_1), ..., Q(s, a_n)]"""

    def __init__(self, state_shape,
                 action_shape,
                 n_quantiles=32,
                 conv_params=None,
                 fc_params=(64, 64),
                 dropout_params=None,
                 dueling=True,
                 noisy_layers=False,
                 activation='relu',
                 init: bool = True,
                 name='QRQNetwork',
                 trainable=True,
                 dtype=tf.float32):
        super().__init__(name=name, trainable=trainable, dtype=dtype)
        self._config = {'state_shape': state_shape,
                        'action_shape': action_shape,
                        'conv_params': conv_params if conv_params else tuple(),
                        'fc_params': fc_params if fc_params else tuple(),
                        'dropout_params': dropout_params if dropout_params else tuple(),
                        'activation': activation,
                        'noisy_layers': noisy_layers,
                        'dueling': dueling,
                        'n_quantiles': n_quantiles,
                        'init': init}
        self._encoder = EncodingNetwork(
            state_shape,
            conv_params=conv_params,
            fc_params=fc_params[:-1],
            dropout_params=dropout_params,
            noisy_layers=noisy_layers,
            activation=activation,
            name=name
        )
        self._q_layer = QLayer(action_shape, units=fc_params[-1], dropout=dropout_params, dueling=dueling,
                               noisy_layers=noisy_layers, n=n_quantiles)
        if init:
            self(tf.ones((1, *state_shape)))

    @property
    def noisy_layers(self):
        return self._encoder.noisy_layers

    @property
    def n_quantiles(self):
        return self._config['n_quantiles']

    def reset_noise(self):
        """Noisy layers' noise reset"""
        self._encoder.reset_noise()
        self._q_layer.reset_noise()

    def call(self, inputs, training=False, mask=None):
        state = self._encoder(inputs, training=training)
        qvals_deltas = self._q_layer(state, training=training)
        # compute qvalues by averaging over quantile dimension
        qvals = tf.reduce_mean(qvals_deltas, axis=2)

        if mask is not None:
            qvals = tf.where(mask, qvals, np.NINF)

        actions = tf.argmax(qvals, axis=1)
        # note that dist params refers to quantiles instead of actions (differently from e.g. vpg and c51)
        return NetworkOutput(critic_values=qvals, actions=actions, dist_params=qvals_deltas)

    def get_config(self):
        config = super().get_config()
        config.update(self._config)
        return config
