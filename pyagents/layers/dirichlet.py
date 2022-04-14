import tensorflow as tf
import tensorflow_probability as tfp


class DirichletLayer(tf.keras.layers.Layer):
    def __init__(self, state_shape, action_shape, name='Dirichlet', dtype=tf.float32, **kwargs):
        super(DirichletLayer, self).__init__(name=name, dtype=dtype, **kwargs)
        self._state_shape = state_shape
        self._action_shape = action_shape
        self._actor_alpha = tf.keras.layers.Dense(*action_shape, activation=tf.math.softplus)

    def call(self, x, training=True):
        alpha = self._actor_alpha(x)
        eps = tf.where(alpha < 0.3, 0.3, 0)
        return alpha + eps
