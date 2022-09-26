import gin
import tensorflow as tf
from pyagents.networks.network import Network, NetworkOutput
from pyagents.networks.policy_network import PolicyNetwork
from pyagents.networks.qnetwork import QNetwork


@gin.configurable
class ACNetwork(Network):
    """
        A network made of a policy network and one (or more) critic networks.
    """

    def __init__(self,
                 state_shape: tuple,
                 action_shape: tuple,
                 pi_params: dict,
                 q_params: dict,
                 pi_out='gaussian',
                 do_init: bool = True,
                 name='ACNetwork',
                 trainable=True,
                 dtype: str = 'float32'):
        super(ACNetwork, self).__init__(name, trainable, dtype)
        self._config = {'state_shape': state_shape,
                        'action_shape': action_shape,
                        'pi_params': pi_params,
                        'q_params': q_params,
                        'pi_out': pi_out,
                        'name': name,
                        'do_init': do_init,
                        'dtype': dtype}
        self._policy_net = PolicyNetwork(state_shape=state_shape,
                                         action_shape=action_shape,
                                         output=pi_out,
                                         **pi_params,
                                         do_init=do_init,
                                         dtype=dtype)
        self._critic_net = QNetwork(state_shape=state_shape,
                                    action_shape=action_shape,
                                    **q_params,
                                    do_init=do_init,
                                    dtype=dtype)

    @property
    def pi(self):
        return self._policy_net

    @property
    def critic(self):
        return self._critic_net

    def get_policy(self):
        return self._policy_net.get_policy()

    def get_config(self):
        config = super().get_config()
        config.update(self._config)
        return config

    def call(self, inputs, training=True, mask=None) -> NetworkOutput:
        pi_out = self._policy_net(inputs, training=training)
        act = pi_out.action
        critic_out = self._critic_net((inputs, act), training=training)
        return NetworkOutput(action=act,
                             dist_params=pi_out.dist_params,
                             critic_values=critic_out.critic_values)
