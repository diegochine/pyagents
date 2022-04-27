import numpy as np
from pyagents.policies.policy import Policy


class RandomPolicy(Policy):

    def __init__(self, state_shape, action_shape):
        super(RandomPolicy, self).__init__(state_shape, action_shape)

    def _act(self, obs, mask=None, training=True):
        return np.array([np.random.randint(self._action_shape)])

    def _distribution(self, obs):
        return np.array([1/self.action_shape] * self.action_shape)
