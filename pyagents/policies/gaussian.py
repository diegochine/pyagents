import gin
import numpy as np
import tensorflow_probability as tfp
import tensorflow as tf
from pyagents.policies.policy import Policy, PolicyOutput
from pyagents.utils import types


@gin.configurable
class GaussianPolicy(Policy):

    def __init__(self,
                 state_shape,
                 action_shape,
                 policy_network,
                 bounds: tuple = (-np.inf, np.inf),
                 dtype=tf.float32):
        super().__init__(state_shape, action_shape)
        self._policy_network = policy_network
        lb, ub = bounds
        self._act_means = tf.constant((ub + lb) / 2.0, shape=action_shape, dtype=dtype)
        self._act_magnitudes = tf.constant((ub - lb) / 2.0, shape=action_shape, dtype=dtype)

    @property
    def is_discrete(self):
        return False

    def _act(self, obs, deterministic=False, mask=None, training=True):
        pi_out = self._policy_network(obs)
        action = pi_out.action.numpy()
        lp = pi_out.logprobs.numpy()
        return PolicyOutput(actions=action, logprobs=lp)

    def entropy(self, output):
        if self._action_shape == (1,):  # orribile
            gaussian = tfp.distributions.Normal(loc=output[0], scale=output[1])
        else:
            gaussian = tfp.distributions.MultivariateNormalDiag(output[0], output[1])
        return gaussian.entropy()

    def log_prob(self, output, actions):
        if self._action_shape == (1,):  # orribile
            gaussian = tfp.distributions.Normal(loc=output[0], scale=output[1])
        else:
            gaussian = tfp.distributions.MultivariateNormalDiag(output[0], output[1])
        actions = tf.math.atanh((actions - self._act_means) / self._act_magnitudes)
        logprobs = tf.reduce_sum(gaussian.log_prob(actions), axis=-1)
        logprobs -= tf.reduce_sum((2. * (tf.math.log(2.) - actions - tf.math.softplus(-2. * actions))), axis=1)
        return logprobs

    def _distribution(self, obs):
        raise NotImplementedError()
