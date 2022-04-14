import gin
import numpy as np
import tensorflow_probability as tfp

from pyagents.policies.policy import Policy
from pyagents.utils import types


@gin.configurable
class GaussianPolicy(Policy):

    def __init__(self,
                 state_shape,
                 action_shape,
                 policy_network,
                 scaling_factor: types.Float = 1.0,
                 bounds: tuple = (-np.inf, np.inf)):
        super().__init__(state_shape, action_shape)
        self._policy_network = policy_network
        self.scaling_factor = scaling_factor
        self._bounds = bounds

    def _act(self, obs, deterministic=True, mask=None, training=True):
        _, (mean, std_dev) = self._policy_network(obs.reshape(1, *obs.shape))
        mean, std_dev = mean.numpy().squeeze(axis=1), std_dev.numpy().squeeze(axis=1)
        if deterministic:
            action = mean
        else:
            action = np.random.normal(loc=mean, scale=std_dev)
        action *= self.scaling_factor
        if self._bounds is not None:
            action = np.clip(action, self._bounds[0], self._bounds[1])
        return action

    def entropy(self, output):
        gaussian = tfp.distributions.Normal(loc=output[0], scale=output[1])
        return gaussian.entropy()

    def log_prob(self, output, actions):
        gaussian = tfp.distributions.Normal(loc=output[0], scale=output[1])
        return gaussian.log_prob(actions)

    def _distribution(self, obs):
        raise NotImplementedError()
