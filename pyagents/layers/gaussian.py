import tensorflow as tf
import tensorflow_probability as tfp


class GaussianLayer(tf.keras.layers.Layer):
    def __init__(self, state_shape, action_shape, deterministic=False, name='Gaussian', dtype=tf.float32, **kwargs):
        super(GaussianLayer, self).__init__(name=name, dtype=dtype, **kwargs)
        self._state_shape = state_shape
        self._action_shape = action_shape
        self._deterministic = deterministic
        self._actor_mean = tf.keras.layers.Dense(action_shape[0],  # assumes 1d action space
                                                 kernel_initializer=tf.keras.initializers.Orthogonal(0.01))
        self._actor_std_dev = tf.keras.layers.Dense(action_shape[0],
                                                    # kernel_initializer=tf.keras.initializers.Orthogonal(0.01),
                                                    activation=tf.math.softplus)

    def call(self, x, training=True):
        mean = self._actor_mean(x)
        std_dev = self._actor_std_dev(x - 0.5) + 0.01
        if self._action_shape == (1,):  # orribile
            mean = tf.squeeze(mean)
            std_dev = tf.squeeze(std_dev)
            gaussian = tfp.distributions.Normal(loc=mean, scale=std_dev)
        else:
            gaussian = tfp.distributions.MultivariateNormalDiag(mean, std_dev)
        if not training or self._deterministic:
            action = mean
        else:
            action = gaussian.sample()
        logprobs = gaussian.log_prob(action)
        if self._action_shape == (1,):  # orribile 2
            action = action[..., tf.newaxis]
            mean = mean[..., tf.newaxis]
            std_dev = std_dev[..., tf.newaxis]
        return action, (mean, std_dev), logprobs
