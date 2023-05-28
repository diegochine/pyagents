import gin
import numpy as np
import tensorflow as tf
from keras.layers import Dense

from pyagents.layers.qlayer import QLayer
from pyagents.networks.network import Network, NetworkOutput
from pyagents.networks.encoding_network import EncodingNetwork


@gin.configurable
class IQNetwork(Network):
    """A DQN for discrete action spaces, taking in input a state and outputting [Q(s, a_1), ..., Q(s, a_n)]"""

    def __init__(self, state_shape,
                 action_shape,
                 conv_params=None,
                 fc_params=(64, 64),
                 dropout_params=None,
                 dueling=True,
                 noisy_layers=False,
                 tau_embedding_size=64,
                 activation='relu',
                 do_init: bool = True,
                 name='IQNetwork',
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
                        'do_init': do_init}
        self._encoder = EncodingNetwork(
            state_shape,
            conv_params=conv_params,
            fc_params=fc_params[:-1],
            dropout_params=dropout_params,
            noisy_layers=noisy_layers,
            activation=activation,
            name=name,
            do_init=do_init
        )
        self._tau_embedding_size = fc_params[-2]  # tau emb shape must match shape of features extracted by the encoder
        self.pis = tf.convert_to_tensor([[[np.pi * i for i in range(self._tau_embedding_size)]]])  # (1, 1, tau_size)
        self._tau_embedding_net = Dense(self._tau_embedding_size, activation='relu')
        self._q_layer = QLayer(action_shape, units=fc_params[-1], dropout=dropout_params, dueling=dueling,
                               noisy_layers=noisy_layers)
        if do_init:
            self(tf.ones((1, *state_shape)))

    @property
    def noisy_layers(self):
        return self._encoder.noisy_layers

    def reset_noise(self):
        """Noisy layers' noise reset"""
        self._encoder.reset_noise()
        self._q_layer.reset_noise()

    def call(self, inputs, training=False, mask=None, num_samples=32):
        batch_size = inputs.shape[0]
        state = self._encoder(inputs, training=training)

        taus = tf.random.uniform((batch_size, num_samples, 1), 0, 1)
        taus_emb = self._tau_embedding_net(tf.cos(self.pis * taus))
        assert taus_emb.shape == (batch_size, num_samples, self._tau_embedding_size)
        state = tf.reshape(tf.expand_dims(state, 1) * taus_emb, (-1, self._tau_embedding_size))
        # the full distribution, at each sampled quantile
        zvals = tf.reshape(self._q_layer(state), (batch_size, num_samples, -1))

        # compute q values, i.e. mean of the distribution (over the sampled quantile dim)
        qvals = tf.reduce_mean(zvals, axis=1)
        if mask is not None:
            qvals = tf.where(mask, qvals, np.NINF)

        actions = tf.argmax(qvals, axis=1)
        # note that dist params refers to quantiles instead of actions (differently from e.g. vpg and c51)
        return NetworkOutput(critic_values=qvals, actions=actions, dist_params=taus)

    def get_config(self):
        config = super().get_config()
        config.update(self._config)
        return config
