import tensorflow as tf
import gin

from pyagents.networks.network import NetworkOutput, Network
from pyagents.networks.qnetwork import QNetwork


@gin.configurable
class ExtendedQNetwork(Network):
    """QNetwork that handles multidimensional rewards and multiple critics (e.g. double clipped q-learning)."""

    def __init__(self, state_shape, action_shape, reward_shape, n_critics=1, critics=None,
                 obs_conv_params=None, obs_fc_params=None, obs_dropout_params=None, obs_activation='relu',
                 act_fc_params=None, act_dropout_params=None, act_activation='relu',
                 fc_params=(64, 64), dropout_params=None, activation='relu',
                 name='ExtendedQNetwork', trainable=True, dtype: str = 'float32', dueling=False):
        super().__init__(name=name, trainable=trainable, dtype=dtype)
        self._config = {'state_shape': state_shape,
                        'action_shape': action_shape,
                        'dueling': dueling,
                        'dtype': dtype,
                        'obs_conv_params': obs_conv_params,
                        'obs_fc_params': obs_fc_params,
                        'obs_dropout_params': obs_dropout_params,
                        'obs_activation': obs_activation,
                        'act_fc_params': act_fc_params,
                        'act_dropout_params': act_dropout_params,
                        'act_activation': act_activation,
                        'fc_params': fc_params,
                        'dropout_params': dropout_params,
                        'activation': activation}

        assert len(reward_shape) == 1 and isinstance(reward_shape[0], int), f'reward shape must be (n,), {reward_shape}'
        self.n_critics = n_critics
        self._reward_shape = reward_shape
        if critics is not None:
            assert isinstance(critics, dict)
            assert all(f'QNet{c}_{r}' in critics for r in range(reward_shape[0]) for c in range(n_critics))
            self._critics = critics
        else:
            self._critics = {f'QNet{c}_{r}': QNetwork(**self._config, name=f'QNet{c}_{r}')
                             for r in range(reward_shape[0])
                             for c in range(n_critics)}
        self._config.update(dict(n_critics=n_critics, reward_shape=reward_shape, name=name))

    @property
    def networks(self):
        return self._critics

    def call(self, inputs, training=False, mask=None):
        """For an input of shape [B, ...], returns an output of shape [B, n_critics, reward_shape]"""
        qvals = tf.stack([tf.concat([self.networks[f'QNet{c}_{r}'](inputs).critic_values
                                     for c in range(self.n_critics)], axis=-1)
                          for r in range(self._reward_shape[0])], axis=-1)
        assert len(qvals.shape) == 3 and qvals.shape[1] == self.n_critics and qvals.shape[2] == self._reward_shape[0]
        return NetworkOutput(critic_values=qvals)

    def get_config(self):
        config = super().get_config()
        config.update(self._config)
        return config
