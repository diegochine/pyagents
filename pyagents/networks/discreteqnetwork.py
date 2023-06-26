import gin
import numpy as np
import torch

from pyagents.layers.qlayer import QLayer
from pyagents.networks.network import Network, NetworkOutput
from pyagents.networks.encoding_network import EncodingNetwork


@gin.configurable
class DiscreteQNetwork(Network):
    """A DQN for discrete action spaces, taking in input a state and outputting [Q(s, a_1), ..., Q(s, a_n)]"""

    def __init__(self, state_shape,
                 action_shape,
                 conv_params=None,
                 fc_params=(64, 64),
                 dropout_params=None,
                 dueling=True,
                 noisy_layers=False,
                 activation='relu',
                 do_init: bool = True,
                 name='QNetwork',
                 trainable=True,
                 dtype: str = 'float32'):
        super().__init__(name=name, trainable=trainable, dtype=dtype)
        self._config = {'state_shape': state_shape,
                        'action_shape': action_shape,
                        'conv_params': conv_params if conv_params else tuple(),
                        'fc_params': fc_params if fc_params else tuple(),
                        'dropout_params': dropout_params if dropout_params else tuple(),
                        'activation': activation,
                        'noisy_layers': noisy_layers,
                        'dueling': dueling,
                        'do_init': do_init,
                        'name': name,
                        'dtype': dtype}
        self._encoder = EncodingNetwork(
            state_shape,
            conv_params=conv_params,
            fc_params=fc_params[:-1],
            dropout_params=dropout_params,
            noisy_layers=noisy_layers,
            activation=activation,
            name=name,
            dtype=dtype,
            do_init=do_init
        )
        self._q_layer = QLayer(action_shape,
                               units=fc_params[-1],
                               dropout=dropout_params,
                               dueling=dueling,
                               noisy_layers=noisy_layers,
                               dtype=dtype)
        if do_init:
            self(torch.ones((1, *state_shape)))

    @property
    def noisy_layers(self):
        return self._encoder.noisy_layers

    def reset_noise(self):
        """Noisy layers' noise reset"""
        self._encoder.reset_noise()
        self._q_layer.reset_noise()

    def forward(self, inputs, training=False, mask=None):
        state = self._encoder(inputs, training=training)
        qvals = self._q_layer(state, training=training)
        if mask is not None:
            qvals = torch.where(mask, qvals, np.NINF)
        action = torch.argmax(qvals, dim=1)
        return NetworkOutput(critic_values=qvals, actions=action)

    def get_config(self):
        config = super().get_config()
        config.update(self._config)
        return config
