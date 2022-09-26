import gin
import tensorflow as tf
from pyagents.networks.network import Network, NetworkOutput
from pyagents.networks.encoding_network import EncodingNetwork
from pyagents.networks.policy_network import PolicyNetwork
from pyagents.networks.value_network import ValueNetwork


@gin.configurable
class SharedBackboneACNetwork(Network):
    """
        A network comprising both a policy head and a critic head, with a shared backbone
    """

    def __init__(self,
                 state_shape,
                 action_shape,
                 output='beta',
                 conv_params=None,
                 fc_params=(64, 64),
                 dropout_params=None,
                 activation='relu',
                 bounds=None,
                 init: bool = True,
                 name='ActorCriticNetwork',
                 trainable=True,
                 dtype: str = 'float32'):
        super(SharedBackboneACNetwork, self).__init__(name, trainable, dtype)
        self._config = {'state_shape': state_shape,
                        'action_shape': action_shape,
                        'conv_params': conv_params if conv_params else [],
                        'fc_params': fc_params if fc_params else [],
                        'dropout_params': dropout_params if dropout_params else [],
                        'activation': activation,
                        'init': init,
                        'name': name,
                        'dtype': dtype,
                        'bounds': bounds,
                        'output': output}
        self._backbone = EncodingNetwork(
            state_shape=state_shape,
            conv_params=conv_params,
            fc_params=fc_params,
            dropout_params=dropout_params,
            activation=activation,
            init=init,
            dtype=dtype
        )
        features_shape = (fc_params[-1],)  # output shape from encoder
        self._policy_head = PolicyNetwork(state_shape=features_shape,
                                          action_shape=action_shape,
                                          output=output,
                                          fc_params=None,
                                          bounds=bounds,
                                          init=init,
                                          dtype=dtype)
        self._critic_head = ValueNetwork(state_shape=features_shape,
                                         fc_params=None,
                                         init=init,
                                         dtype=dtype)

    def get_policy(self):
        return self._policy_head.get_policy(caller=self)

    def get_config(self):
        config = super().get_config()
        config.update(self._config)
        return config

    def call(self, inputs, training=True, mask=None) -> NetworkOutput:
        state = self._backbone(inputs, training=training)
        pi_out = self._policy_head(state)
        critic_out = self._critic_head(state)
        return NetworkOutput(actions=pi_out.action,
                             dist_params=pi_out.dist_params,
                             critic_values=critic_out.critic_values)
