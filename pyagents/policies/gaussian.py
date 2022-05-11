import gin
import numpy as np
import tensorflow_probability as tfp

from pyagents.policies.policy import Policy, PolicyOutput
from pyagents.utils import types


@gin.configurable
class GaussianPolicy(Policy):

    def __init__(self,
                 state_shape,
                 action_shape,
                 policy_network,
                 bounds: tuple = (-np.inf, np.inf)):
        super().__init__(state_shape, action_shape)
        self._policy_network = policy_network

    @property
    def is_discrete(self):
        return False

    def _act(self, obs, deterministic=False, mask=None, training=True):
        pi_out = self._policy_network(obs)
        action = pi_out.action.numpy()
        lp = pi_out.logprobs.numpy()
        return PolicyOutput(actions=action, logprobs=lp)

    def entropy(self, output):
        if self._action_shape == (1,):  # orribile
            gaussian = tfp.distributions.Normal(loc=output[0], scale=output[1])
        else:
            gaussian = tfp.distributions.MultivariateNormalDiag(output[0], output[1])
        return gaussian.entropy()

    def log_prob(self, output, actions):
        if self._action_shape == (1,):  # orribile
            gaussian = tfp.distributions.Normal(loc=output[0], scale=output[1])
        else:
            gaussian = tfp.distributions.MultivariateNormalDiag(output[0], output[1])
        return gaussian.log_prob(actions)

    def _distribution(self, obs):
        raise NotImplementedError()
