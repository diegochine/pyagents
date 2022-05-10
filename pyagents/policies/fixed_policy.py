import numpy as np
import gin

from pyagents.policies.policy import Policy, PolicyOutput
from pyagents.utils import types


@gin.configurable
class FixedPolicy(Policy):
    
    def __init__(self,
                 state_shape,
                 action_shape,
                 policy_network,
                 bounds: tuple = None):
        super(FixedPolicy, self).__init__(state_shape, action_shape)
        self._policy_network = policy_network
        self.bounds = bounds

    @property
    def is_discrete(self):
        return False
    
    def _act(self, obs, **kwargs):
        action = self._policy_network(obs, **kwargs).action
        if self.bounds is not None:
            action = np.clip(action, self.bounds[0], self.bounds[1])
        return PolicyOutput(actions=action)

    def log_prob(self, output, action):
        return self._policy.log_prob(output, action)

    def entropy(self, output):
        raise NotImplementedError('Entropy not available with eps-greedy')

    @property
    def is_discrete(self):
        return True  # TODO should allow for continuous policies as well
