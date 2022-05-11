import numpy as np
from pyagents.policies.policy import Policy, PolicyOutput


class RandomDiscretePolicy(Policy):

    def __init__(self, state_shape, action_shape):
        super(RandomDiscretePolicy, self).__init__(state_shape, action_shape)

    @property
    def is_discrete(self):
        return True

    def _act(self, obs, mask=None, training=True):
        actions = np.array([np.random.randint(self._action_shape)
                            for _ in range(obs.shape[0])])  # batch dimension
        return PolicyOutput(actions=actions)

    def _distribution(self, obs):
        return np.array([1/self.action_shape] * self.action_shape)
