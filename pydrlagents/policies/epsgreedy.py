import gin
import numpy as np
from policies.policy import Policy
from policies.randompolicy import RandomPolicy

@gin.configurable
class EpsGreedyPolicy(Policy):

    def __init__(self, policy, epsilon=0.1, epsilon_decay=0.99, epsilon_min=0.01):
        self._policy = policy
        self._random_policy = RandomPolicy(self._policy.state_shape, self._policy.action_shape)
        self._epsilon = epsilon
        self._epsilon_decay = epsilon_decay
        self._epsilon_min = epsilon_min
        super().__init__(self._policy.state_shape, self._policy.action_shape)

    def update_eps(self):
        self._epsilon = max(self._epsilon_min, self._epsilon * self._epsilon_decay)

    @property
    def epsilon(self):
        return self._epsilon

    def _act(self, obs):
        if np.random.rand() <= self._epsilon:
            return self._random_policy.act(obs)
        else:
            return self._policy.act(obs)

    def _distribution(self, obs):
        raise NotImplementedError()
