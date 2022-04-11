import gin
import tensorflow as tf

from pyagents.layers.qlayer import QLayer
from pyagents.networks.network import Network
from pyagents.networks.encoding_network import EncodingNetwork


@gin.configurable
class QNetwork(Network):
    """A generic DQN for both discrete and continuous action spaces, taking in input a (state, action) pair
     and outputting Q(state, action) """

    def __init__(self, state_shape, action_shape,
                 obs_conv_params=None, obs_fc_params=(64, 64), obs_dropout_params=None, obs_activation='relu',
                 act_fc_params=None, act_dropout_params=None, act_activation='relu',
                 fc_params=(64, 64), dropout_params=None, activation='relu',
                 name='QNetwork', trainable=True, dtype=tf.float32, dueling=False):
        super().__init__(name=name, trainable=trainable, dtype=dtype)
        self._config = {'state_shape': state_shape,
                        'action_shape': action_shape,
                        'dueling': dueling,
                        'name': name}
        self._obs_encoder = EncodingNetwork(
            state_shape,
            conv_params=obs_conv_params,
            fc_params=obs_fc_params,
            dropout_params=obs_dropout_params,
            activation=obs_activation,
            name="obs_encoder"
        )
        self._config['obs_encoder'] = self._obs_encoder.get_config()
        if act_fc_params is not None:
            self._act_encoder = EncodingNetwork(
                state_shape,
                conv_params=None,
                fc_params=act_fc_params,
                dropout_params=act_dropout_params,
                activation=act_activation,
                name="act_encoder")
            self._config['act_encoder'] = self._act_encoder.get_config()
        else:
            self._act_encoder = None
            self._config['act_encoder'] = {}
        self._shared_encoder = EncodingNetwork(
            state_shape,
            conv_params=None,
            fc_params=fc_params[:-1],
            dropout_params=dropout_params,
            activation=activation,
            name="shared_encoder"
        )
        self._q_layer = QLayer(1, units=fc_params[-1], dropout=dropout_params, dueling=dueling)
        self._config['shared_encoder'] = self._shared_encoder.get_config()

    def call(self, inputs, training=False, mask=None):
        obs, action = inputs
        state = self._obs_encoder(obs, training=training)
        if self._act_encoder is not None:
            action = self._act_encoder(action, training=training)
        state_action = tf.concat([state, action], axis=1)
        q_values = self._q_layer(state_action, training=training)
        return q_values

    def get_config(self):
        config = super().get_config()
        config.update(self._config)
        return config
