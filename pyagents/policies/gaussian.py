import gin
import numpy as np
import tensorflow_probability as tfp
import tensorflow as tf
from pyagents.policies.policy import Policy, PolicyOutput


@gin.configurable
class GaussianPolicy(Policy):

    def __init__(self,
                 state_shape,
                 action_shape,
                 policy_network,
                 bounds: tuple = (-np.inf, np.inf)):
        super().__init__(state_shape, action_shape)
        self._policy_network = policy_network
        self._bounds = bounds

    @property
    def is_discrete(self):
        return False

    @property
    def bounds(self):
        return self._bounds

    def _act(self, obs, deterministic=False, mask=None, training=True):
        pi_out = self._policy_network(obs, training=training)
        action = pi_out.actions.numpy()
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
