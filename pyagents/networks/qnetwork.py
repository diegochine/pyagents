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
                 obs_conv_params=None, obs_fc_params=(64, 64), obs_dropout_params=None, obs_activation='relu',
                 act_fc_params=None, act_dropout_params=None, act_activation='relu',
                 fc_params=(64, 64), dropout_params=None, activation='relu',
                 name='QNetwork', trainable=True, dtype: str = 'float32', dueling=False):
        super().__init__(name=name, trainable=trainable, dtype=dtype)
        self._config = {'state_shape': state_shape,
                        'action_shape': action_shape,
                        'dueling': dueling,
                        'name': name,
                        'dtype': dtype,
                        'obs_conv_params': obs_conv_params if obs_conv_params else [],
                        'obs_fc_params': obs_fc_params if obs_fc_params else [],
                        'obs_dropout_params': obs_dropout_params if obs_dropout_params else [],
                        'obs_activation': obs_activation,
                        'act_fc_params': act_fc_params if act_fc_params else [],
                        'act_dropout_params': act_dropout_params if act_dropout_params else [],
                        'act_activation': act_activation,
                        'fc_params': fc_params if fc_params else [],
                        'dropout_params': dropout_params if dropout_params else [],
                        'activation': activation}
        self._obs_encoder = EncodingNetwork(
            state_shape,
            conv_params=obs_conv_params,
            fc_params=obs_fc_params,
            dropout_params=obs_dropout_params,
            activation=obs_activation,
            name="obs_encoder",
            dtype=dtype
        )
        if act_fc_params is not None:
            self._act_encoder = EncodingNetwork(
                action_shape,
                conv_params=None,
                fc_params=act_fc_params,
                dropout_params=act_dropout_params,
                activation=act_activation,
                name="act_encoder",
                dtype=dtype)
        else:
            self._act_encoder = None

        self._shared_encoder = EncodingNetwork(
            (obs_fc_params[-1] + (action_shape[0] if act_fc_params is None or len(act_fc_params) == 0 else act_fc_params[-1]),),
            conv_params=None,
            fc_params=fc_params[:-1],
            dropout_params=dropout_params,
            activation=activation,
            name="shared_encoder",
            dtype=dtype
        )
        self._q_layer = QLayer(1, units=fc_params[-1], dropout=dropout_params, dueling=dueling, dtype=dtype)
        self((tf.ones((1, *state_shape)), tf.ones((1, *action_shape))))

    def call(self, inputs, training=False, mask=None):
        obs, action = inputs
        state = self._obs_encoder(obs, training=training)
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
