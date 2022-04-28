import numpy as np
import gin

from pyagents.policies.policy import Policy
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
        return action
