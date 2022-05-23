import gin
import numpy as np
import tensorflow as tf

from pyagents.layers.qlayer import QLayer
from pyagents.networks.network import Network, NetworkOutput
from pyagents.networks.encoding_network import EncodingNetwork


@gin.configurable
class C51QNetwork(Network):
    """A DQN for discrete action spaces, taking in input a state and outputting [Q(s, a_1), ..., Q(s, a_n)]"""

    def __init__(self, state_shape,
                 action_shape,
                 n_atoms=51,
                 conv_params=None,
                 fc_params=(64, 64),
                 dropout_params=None,
                 dueling=True,
                 noisy_layers=False,
                 support=None,
                 activation='relu',
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
                        'n_atoms': n_atoms}
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
                               noisy_layers=noisy_layers, n=n_atoms)
        self._support = support
        if support is not None:  # loading model
            self(tf.ones((1, *state_shape)))

    @property
    def noisy_layers(self):
        return self._encoder.noisy_layers

    @property
    def n_atoms(self):
        return self._config['n_atoms']

    def set_support(self, support):
        self._support = support
        self._config['support'] = support

    def reset_noise(self):
        """Noisy layers' noise reset"""
        self._encoder.reset_noise()
        self._q_layer.reset_noise()

    def call(self, inputs, training=False, mask=None):
        assert self._support is not None, 'must first call set_support() method before forward of distributional qnet'
        state = self._encoder(inputs, training=training)
        qvals_logits = self._q_layer(state, training=training)
        # normalize returns distribution
        qvals_dist = tf.nn.softmax(qvals_logits, axis=2)
        # compute qvalues as z^T · p(s, a | theta)
        qvals = tf.tensordot(qvals_dist, self._support, axes=1)

        if mask is not None:
            qvals = tf.where(mask, qvals, np.NINF)

        actions = tf.argmax(qvals, axis=1)
        # note that dist params refers to values instead of actions (differently from e.g. vpg and ppo)
        return NetworkOutput(critic_values=qvals, actions=actions, logits=qvals_logits, dist_params=qvals_dist)

    def get_config(self):
        config = super().get_config()
        config.update(self._config)
        return config
