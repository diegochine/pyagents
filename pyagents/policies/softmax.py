import numpy as np
import tensorflow_probability as tfp
from pyagents.policies.policy import Policy


class SoftmaxPolicy(Policy):

    def __init__(self, state_shape, action_shape, actor_network):
        super().__init__(state_shape, action_shape)
        self._actor_network = actor_network

    def _act(self, obs, deterministic=True, mask=None, training=True):
        probs = self._actor_network(obs.reshape(1, *obs.shape))
        probs = probs.numpy().squeeze(axis=0)
        if deterministic:
            return np.argmax(probs)
        else:
            return tfp.distributions.Categorical(probs=probs).sample().numpy()

    def entropy(self, output):
        gaussian = tfp.distributions.Categorical(probs=output)
        return gaussian.entropy()

    def log_prob(self, output, actions):
        dist = tfp.distributions.Categorical(probs=output)
        return dist.log_prob(actions)

    def _distribution(self, obs):
        raise NotImplementedError()
