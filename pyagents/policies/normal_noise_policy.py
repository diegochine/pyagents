from typing import Optional

import gin
import tensorflow as tf
from pyagents.policies import Policy
from pyagents.policies.policy import PolicyOutput


@gin.configurable
class NormalNoisePolicy(Policy):

    def __init__(self, policy: Policy,
                 stddev: float = 0.1,
                 decay: Optional[float] = None,
                 stddev_min: Optional[float] = None,
                 bounds: tuple = None):
        self._policy = policy
        self._stddev = stddev
        self._decay = decay
        self._stddev_min = stddev_min
        self._bounds = bounds
        super().__init__(self._policy.state_shape, self._policy.action_shape)

    @property
    def is_discrete(self):
        return self._policy.is_discrete

    @property
    def bounds(self):
        return self._bounds

    def _act(self, obs, mask=None, training=True, **kwargs):
        action = self._policy.act(obs, mask=mask, training=training).actions
        if training:
            eps = tf.random.normal(self._policy.action_shape, mean=0.0, stddev=self._stddev)
            action += eps
        if self._bounds is not None:
            action = tf.clip_by_value(action, self._bounds[0], self._bounds[1])
        if self._decay is not None:
            self._stddev = min(self._stddev * self._decay, self._stddev_min)
        return PolicyOutput(actions=action)
