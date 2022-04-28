import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from pyagents.policies.policy import Policy


class SoftmaxPolicy(Policy):

    @property
    def is_discrete(self) -> bool:
        return True

    def __init__(self, state_shape, action_shape, policy_network):
        super().__init__(state_shape, action_shape)
        self._policy_network = policy_network

    def _act(self, obs, deterministic=False, mask=None, training=True):
        pi_out = self._policy_network(obs)
        probs = pi_out.dist_params
        probs = probs.numpy()
        if deterministic:
            return np.argmax(probs, axis=0)
        else:
            return tfp.distributions.Categorical(probs=probs).sample().numpy()

    def entropy(self, output):
        dist = tfp.distributions.Categorical(probs=output)
        return dist.entropy()

    def log_prob(self, output, actions):
        dist = tfp.distributions.Categorical(probs=output)
        return dist.log_prob(tf.squeeze(actions))

    def _distribution(self, obs):
        raise NotImplementedError()
