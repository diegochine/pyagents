import gin
import tensorflow as tf

from pyagents.layers.qlayer import QLayer
from pyagents.networks.network import Network, NetworkOutput
from pyagents.networks.encoding_network import EncodingNetwork


@gin.configurable
class QNetwork(Network):
    """A generic DQN for both discrete and continuous action spaces, taking in input a (state, action) pair
     and outputting Q(state, action) """

    def __init__(self, state_shape, action_shape,
                 obs_conv_params=None, obs_fc_params=None, obs_dropout_params=None, obs_activation='relu',
                 act_fc_params=None, act_dropout_params=None, act_activation='relu',
                 fc_params=(64, 64), dropout_params=None, activation='relu',
                 init: bool = True, name='QNetwork', trainable=True, dtype: str = 'float32', dueling=False):
        super().__init__(name=name, trainable=trainable, dtype=dtype)
        self._config = {'state_shape': state_shape,
                        'action_shape': action_shape,
                        'dueling': dueling,
                        'name': name,
                        'dtype': dtype,
                        'init': init,
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

        if obs_conv_params is None and obs_fc_params is None:
            self._obs_encoder = None
            shared_obs_input = state_shape[0]  # let's assume that if there's no encoding there's one-dim space
        else:
            self._obs_encoder = EncodingNetwork(
                state_shape,
                conv_params=obs_conv_params,
                fc_params=obs_fc_params,
                dropout_params=obs_dropout_params,
                activation=obs_activation,
                name="obs_encoder",
                init=init,
                dtype=dtype
            )
            shared_obs_input = obs_fc_params[-1]

        if act_fc_params is None:
            self._act_encoder = None
            shared_act_input = action_shape[0]  # let's assume that if there's no encoding there's one-dim space
        else:
            self._act_encoder = EncodingNetwork(
                action_shape,
                conv_params=None,
                fc_params=act_fc_params,
                dropout_params=act_dropout_params,
                activation=act_activation,
                name="act_encoder",
                init=init,
                dtype=dtype)
            shared_act_input = act_fc_params[-1]

        self._shared_encoder = EncodingNetwork(
            (shared_obs_input + shared_act_input,),
            conv_params=None,
            fc_params=fc_params[:-1],
            dropout_params=dropout_params,
            activation=activation,
            name="shared_encoder",
            init=init,
            dtype=dtype
        )
        self._q_layer = QLayer(1, units=fc_params[-1], dropout=dropout_params, dueling=dueling, dtype=dtype)
        if init:
            self((tf.ones((1, *state_shape)), tf.ones((1, *action_shape))))

    def call(self, inputs, training=False, mask=None):
        state, action = inputs
        if self._obs_encoder is not None:
            state = self._obs_encoder(state, training=training)
        if self._act_encoder is not None:
            action = self._act_encoder(action, training=training)
        state_action = tf.concat([state, action], axis=1)
        state_action = self._shared_encoder(state_action, training=training)
        q_values = self._q_layer(state_action, training=training)
        return NetworkOutput(critic_values=q_values)

    def get_config(self):
        config = super().get_config()
        config.update(self._config)
        return config
