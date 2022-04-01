import numpy as np
import tensorflow_probability as tfp
from pyagents.policies.policy import Policy


class GaussianPolicy(Policy):

    def __init__(self, state_shape, action_shape, actor_network):
        super().__init__(state_shape, action_shape)
        self._actor_network = actor_network

    def _act(self, obs, deterministic=True, mask=None, training=True):
        mean, std_dev = self._actor_network(obs.reshape(1, *obs.shape))
        mean, std_dev = mean.numpy().squeeze(axis=1), std_dev.numpy().squeeze(axis=1)
        if deterministic:
            return mean
        else:
            return np.random.normal(loc=mean, scale=std_dev)

    def entropy(self, output):
        gaussian = tfp.distributions.Normal(loc=output[0], scale=output[1])
        return gaussian.entropy()

    def log_prob(self, output, actions):
        gaussian = tfp.distributions.Normal(loc=output[0], scale=output[1])
        return gaussian.log_prob(actions)

    def _distribution(self, obs):
        raise NotImplementedError()
