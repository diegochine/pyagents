import tensorflow as tf
import tensorflow_probability as tfp


class SoftmaxLayer(tf.keras.layers.Layer):
    def __init__(self, state_shape, action_shape, name='Softmax', dtype=tf.float32, **kwargs):
        super(SoftmaxLayer, self).__init__(name=name, dtype=dtype, **kwargs)
        self._state_shape = state_shape
        self._action_shape = action_shape
        self._preferences = tf.keras.layers.Dense(action_shape[0], activation='linear',  # assumes 1d action space
                                                  kernel_initializer=tf.keras.initializers.Orthogonal(0.01))

    def call(self, x, training=True):
        preferences = self._preferences(x)  # logits
        dist = tfp.distributions.Categorical(logits=preferences)
        if training:  # TODO maybe not ok for all algos see ddpg
            action = dist.sample()
        else:
            action = tf.math.argmax(preferences, axis=1)
        logprobs = dist.log_prob(action)
        return action, preferences, logprobs
