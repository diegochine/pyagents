import gin
import tensorflow as tf
from pyagents.policies import Policy


@gin.configurable
class NormalNoisePolicy(Policy):

    def __init__(self, policy: Policy, stddev: float = 0.1, bounds: tuple = None):
        self._policy = policy
        self._stddev = stddev
        self._bounds = bounds
        super().__init__(self._policy.state_shape, self._policy.action_shape)

    def _act(self, obs, mask=None, training=True, **kwargs):
        action = self._policy.act(obs, mask=mask, training=training)
        if training:
            eps = tf.random.normal(self._policy.action_shape, mean=0.0, stddev=self._stddev)
            action += eps
        if self._bounds is not None:
            action = tf.clip_by_value(action, self._bounds[0], self._bounds[1])
        return action