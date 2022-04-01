import tensorflow as tf
import tensorflow_probability as tfp


class GaussianLayer(tf.keras.layers.Layer):
    def __init__(self, state_shape, action_shape, **kwargs):
        super(GaussianLayer, self).__init__(**kwargs)
        self._state_shape = state_shape
        self._action_shape = action_shape
        self._actor_mean = tf.keras.layers.Dense(action_shape[0])  # assumes 1d action space
        self._actor_std_dev = tf.keras.layers.Dense(action_shape[0], activation=tf.math.softplus)

    def call(self, x, training=True):
        mean = self._actor_mean(x)
        std_dev = self._actor_std_dev(x)
        return mean, std_dev
