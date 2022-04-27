import numpy as np
import tensorflow_probability as tfp

from pyagents.policies.policy import Policy


class SoftmaxPolicy(Policy):

    def __init__(self, state_shape, action_shape, policy_network):
        super().__init__(state_shape, action_shape)
        self._policy_network = policy_network

    def _act(self, obs, deterministic=False, mask=None, training=True):
        pi_out = self._policy_network(obs.reshape(1, *obs.shape))
        probs = pi_out.dist_params
        probs = probs.numpy().squeeze(axis=0)
        if deterministic:
            return np.argmax(probs)
        else:
            return tfp.distributions.Categorical(probs=probs).sample().numpy()

    def entropy(self, output):
        dist = tfp.distributions.Categorical(probs=output)
        return dist.entropy()

    def log_prob(self, output, actions):
        dist = tfp.distributions.Categorical(probs=output)
        return dist.log_prob(actions)

    def _distribution(self, obs):
        raise NotImplementedError()
