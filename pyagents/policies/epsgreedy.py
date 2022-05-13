import gin
import numpy as np
from pyagents.policies.policy import Policy
from pyagents.policies.randomdiscretepolicy import RandomDiscretePolicy


@gin.configurable
class EpsGreedyPolicy(Policy):

    def __init__(self, policy, epsilon=0.1, epsilon_decay=0.99, epsilon_min=0.01):
        self._policy = policy
        self._random_policy = RandomDiscretePolicy(self._policy.state_shape, self._policy.action_shape)
        self._epsilon = epsilon
        self._epsilon_decay = epsilon_decay
        self._epsilon_min = epsilon_min
        super().__init__(self._policy.state_shape, self._policy.action_shape)

    def update_eps(self):
        self._epsilon = max(self._epsilon_min, self._epsilon * self._epsilon_decay)

    @property
    def is_discrete(self):
        return True  # TODO should also work for continuous policies

    def log_prob(self, output, action):
        raise NotImplementedError('method not exposed by eps greedy')

    def entropy(self, output):
        raise NotImplementedError('method not exposed by eps greedy')


    @property
    def epsilon(self):
        return self._epsilon

    def _act(self, obs, mask=None, training=True):
        if training and np.random.rand() <= self._epsilon:
            return self._random_policy.act(obs, mask=mask, training=training)
        else:
            return self._policy.act(obs, mask=mask, training=training)

    def _distribution(self, obs):
        if np.random.rand() <= self._epsilon:
            return self._random_policy.distribution(obs)
        else:
            return self._policy.distribution(obs)
