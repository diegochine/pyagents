import tensorflow as tf
import tensorflow_probability as tfp


class GaussianLayer(tf.keras.layers.Layer):
    def __init__(self, state_shape, action_shape, bounds, deterministic=False, start_std=0.5, std_eps=0.01,
                 name='Gaussian', dtype=tf.float32, **kwargs):
        super(GaussianLayer, self).__init__(name=name, dtype=dtype, **kwargs)
        self._state_shape = state_shape
        self._action_shape = action_shape
        self._deterministic = deterministic
        # determine parameters for rescaling
        lb, ub = bounds
        self._act_means = tf.constant((ub + lb) / 2.0, shape=action_shape, dtype=dtype)
        self._act_magnitudes = tf.constant((ub - lb) / 2.0, shape=action_shape, dtype=dtype)
        self._mean_layer = tf.keras.layers.Dense(action_shape[0],  # assumes 1d action space
                                                 kernel_initializer=tf.keras.initializers.Orthogonal(0.01))
        # initialize std dev variable(s)
        std_init = tfp.math.softplus_inverse(start_std - std_eps)
        self._std_dev = tf.Variable(initial_value=tf.fill(action_shape, std_init),
                                    shape=self._action_shape,
                                    trainable=True)
        self._std_eps = std_eps

    def call(self, x, training=True):
        mean = self._mean_layer(x)
        std_dev = tf.math.softplus(self._std_dev) + self._std_eps
        if self._action_shape == (1,):
            mean = tf.squeeze(mean, axis=1)
            gaussian = tfp.distributions.Normal(loc=mean, scale=std_dev)
        else:
            gaussian = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=std_dev)
        if not training or self._deterministic:
            action = mean
        else:
            action = gaussian.sample()
        if self._action_shape == (1,):  # orribile
            action = action[..., tf.newaxis]
            mean = mean[..., tf.newaxis]
            std_dev = std_dev[..., tf.newaxis]

        logprobs = tf.reduce_sum(gaussian.log_prob(action), axis=-1)
        logprobs -= tf.reduce_sum((2. * (tf.math.log(2.) - action - tf.math.softplus(-2. * action))), axis=1)

        action = self._act_means + self._act_magnitudes * tf.math.tanh(action)
        return action, (mean, std_dev), logprobs
