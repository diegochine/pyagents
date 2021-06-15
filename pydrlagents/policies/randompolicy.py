import numpy as np
from policies.policy import Policy


class RandomPolicy(Policy):

    def __init__(self, state_shape, action_shape):
        super(RandomPolicy, self).__init__(state_shape, action_shape)

    def _act(self, obs):
        return np.random.randint(self._action_shape)

    def _distribution(self, obs):
        raise NotImplementedError()
