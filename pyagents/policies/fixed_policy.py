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
                 scaling_factor: types.Float = 1.0,
                 bounds: tuple = None):
        super(FixedPolicy, self).__init__(state_shape, action_shape)
        self._policy_network = policy_network
        self.scaling_factor = scaling_factor
        self.bounds = bounds
    
    def _act(self, obs, **kwargs):
        a = self._policy_network(obs) * self.scaling_factor
        if self.bounds is not None:
            a = np.clip(a, self.bounds[0], self.bounds[1])
        return a
