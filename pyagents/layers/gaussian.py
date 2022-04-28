import tensorflow as tf
import tensorflow_probability as tfp


class GaussianLayer(tf.keras.layers.Layer):
    def __init__(self, state_shape, action_shape, deterministic=False, name='Gaussian', dtype=tf.float32, **kwargs):
        super(GaussianLayer, self).__init__(name=name, dtype=dtype, **kwargs)
        self._state_shape = state_shape
        self._action_shape = action_shape
        self._deterministic = deterministic
        self._actor_mean = tf.keras.layers.Dense(action_shape[0],
                                                 kernel_initializer=tf.random_uniform_initializer(minval=-0.003, maxval=0.003))  # assumes 1d action space
        self._actor_std_dev = tf.keras.layers.Dense(action_shape[0],
                                                    activation=tf.math.softplus)

    def call(self, x, training=True):
        mean = self._actor_mean(x)
        std_dev = self._actor_std_dev(x - 0.4) + 0.01
        if not training or self._deterministic:
            action = mean
        else:
            gaussian = tfp.distributions.Normal(loc=mean, scale=std_dev)
            action = gaussian.sample()
        return action, (mean, std_dev)
