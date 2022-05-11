import numpy as np
import tensorflow as tf
from pyagents.policies.policy import Policy, PolicyOutput


class QPolicy(Policy):

    def __init__(self, state_shape, action_shape, q_network):
        super().__init__(state_shape, action_shape)
        self._q_network = q_network

    @property
    def is_discrete(self):
        return True

    def _act(self, obs, mask=None, training=True):
        qvals = self._q_network(obs)
        if mask is not None:
            assert isinstance(mask, np.ndarray)
            qvals = tf.where(mask, qvals, np.NINF)
        return PolicyOutput(actions=tf.argmax(qvals, axis=1).numpy())

    def _distribution(self, obs):
        qvals = self._q_network(obs.reshape(1, *obs.shape))
        return qvals / tf.reduce_sum(qvals)
