import gin
import tensorflow as tf
from pyagents.networks.network import Network
from pyagents.networks.encoding_network import EncodingNetwork
from pyagents.networks.policy_network import PolicyNetwork
from pyagents.networks.value_network import ValueNetwork


@gin.configurable
class ActorCriticNetwork(Network):
    """
        A network comprising both a policy head and a critic head, with a shared backbone
    """

    def __init__(self,
                 state_shape,
                 action_shape,
                 distribution='beta',
                 preprocessing_layers=None,
                 conv_layer_params=None,
                 fc_layer_params=(64, 64),
                 dropout_params=None,
                 activation='relu',
                 name='ActorNetwork',
                 trainable=True,
                 dtype=tf.float32):
        super(ActorCriticNetwork, self).__init__(name, trainable, dtype)
        self._config = {'state_shape': state_shape,
                        'action_shape': action_shape,
                        'preprocessing_layers': [lay.config() for lay in preprocessing_layers]
                        if preprocessing_layers else [],
                        'conv_layer_params': conv_layer_params if conv_layer_params else [],
                        'fc_layer_params': fc_layer_params if fc_layer_params else [],
                        'dropout_params': dropout_params if dropout_params else [],
                        'activation': activation,
                        'name': name}
        self._encoder = EncodingNetwork(
            state_shape,
            preprocessing_layers=preprocessing_layers,
            conv_layer_params=conv_layer_params,
            fc_layer_params=fc_layer_params,
            dropout_params=dropout_params,
            activation=activation,
            dtype=dtype
        )
        features_shape = (fc_layer_params[-1],)  # output shape from encoder
        self._policy_head = PolicyNetwork(state_shape=features_shape,
                                          action_shape=action_shape,
                                          distribution=distribution,
                                          fc_layer_params=None,
                                          dtype=dtype)
        self._critic_head = ValueNetwork(state_shape=features_shape,
                                         fc_layer_params=None,
                                         dtype=dtype)

    def get_policy(self):
        return self._policy_head.get_policy(caller=self)

    def get_config(self):
        config = super().get_config()
        config.update(self._config)
        return config

    def call(self, inputs, training=True, mask=None):
        state = self._encoder(inputs, training=training)
        dist_params = self._policy_head(state)
        critic_value = self._critic_head(state)
        return dist_params, critic_value
