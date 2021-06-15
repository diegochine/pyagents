import numpy as np
from policies.policy import Policy


class QPolicy(Policy):

    def __init__(self, state_shape, action_shape, q_network):
        super().__init__(state_shape, action_shape)
        self._q_network = q_network

    def _act(self, obs):
        qvals = self._q_network(obs.reshape(1, *obs.shape))
        return np.argmax(qvals)

    def _distribution(self, obs):
        raise NotImplementedError()
