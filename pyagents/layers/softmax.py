import tensorflow as tf


class SoftmaxLayer(tf.keras.layers.Layer):
    def __init__(self, state_shape, action_shape, name='Softmax', dtype=tf.float32, **kwargs):
        super(SoftmaxLayer, self).__init__(name=name, dtype=dtype, **kwargs)
        self._state_shape = state_shape
        self._action_shape = action_shape
        self._preferences = tf.keras.layers.Dense(action_shape[0], activation='linear')  # assumes 1d action space
        self._softmax = tf.nn.softmax

    def call(self, x, training=True):
        preferences = self._preferences(x)
        probs = self._softmax(preferences)
        action = tf.math.argmax(probs)
        return action, probs
