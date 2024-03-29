import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class GaussianLayer(tf.keras.layers.Layer):
    def __init__(self, state_shape, action_shape, bounds, deterministic=False, mean_activation='tanh',
                 start_std=0.5, std_eps=0.01, state_dependent_std=True, name='Gaussian', dtype=tf.float32, **kwargs):
        super(GaussianLayer, self).__init__(name=name, dtype=dtype, **kwargs)
        self._state_shape = state_shape
        self._action_shape = action_shape
        self._deterministic = deterministic
        self._state_dependent_std = state_dependent_std
        # determine parameters for rescaling
        lb, ub = bounds
        self._act_means = tf.constant((ub + lb) / 2.0, shape=action_shape, dtype=dtype)
        self._act_magnitudes = tf.constant((ub - lb) / 2.0, shape=action_shape, dtype=dtype)
        self._mean_layer = tf.keras.layers.Dense(action_shape[0],  # assumes 1d action space
                                                 kernel_initializer=tf.keras.initializers.Orthogonal(0.01),
                                                 activation=mean_activation)
        # initialize std dev variable(s)
        std_init = tfp.math.softplus_inverse(start_std - std_eps)
        if self._state_dependent_std:
            self._std_dev = tf.keras.layers.Dense(action_shape[0],
                                                  kernel_initializer=tf.keras.initializers.Orthogonal(std_init),
                                                  activation=None)
        else:
            self._std_dev = tf.Variable(initial_value=tf.fill(action_shape, std_init),
                                        shape=self._action_shape,
                                        trainable=True)
            self._std_eps = std_eps

    def call(self, x, training=True):
        mean = self._mean_layer(x)
        if self._state_dependent_std:
            std_dev = tf.exp(self._std_dev(x) + 1e-5)
        else:
            mean = self._act_means + self._act_magnitudes * mean
            std_dev = tf.math.softplus(self._std_dev) + self._std_eps

        gaussian = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(mean.shape), scale_diag=tf.ones(std_dev.shape))

        if not training or self._deterministic:
            action = tfp.bijectors.Tanh()(mean)
            log_probs = tf.fill(
                tf.concat((tf.shape(mean)[:-1], [1]), axis=0), np.inf)
        else:
            raw_action_distribution = tfp.bijectors.Chain((
                tfp.bijectors.Shift(mean),
                tfp.bijectors.Scale(std_dev),
            ))(gaussian)
            action_distribution = tfp.bijectors.Tanh()(raw_action_distribution)
            action = action_distribution.sample()
            log_probs = action_distribution.log_prob(action)

        # action rescaling into bounds
        action = self._act_means + self._act_magnitudes * action

        # if not training or self._deterministic:
        #     action = mean
        # else:
        #     action = gaussian.sample()
        # logprobs = gaussian.log_prob(action)
        #if self._action_shape == (1, ):  # orribile
        #    action = action[..., tf.newaxis]
        #    mean = mean[..., tf.newaxis]
        #    std_dev = std_dev[..., tf.newaxis]
        #
        # if self._state_dependent_std:
        #     # logprobs = tf.reduce_sum(tf.expand_dims(logprobs, 1), axis=-1)
        #     logprobs -= tf.reduce_sum((2. * (tf.math.log(2.) - action - tf.math.softplus(-2. * action))), axis=-1)
        #     action = self._act_means + self._act_magnitudes * tf.math.tanh(action)

        return action, (mean, std_dev), log_probs
