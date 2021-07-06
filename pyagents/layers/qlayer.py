import gin
import tensorflow as tf


@gin.configurable
class QLayer(tf.keras.layers.Layer):

    def __init__(self, action_shape, units=32, dueling=True, **kwargs):
        super().__init__(**kwargs)
        self.action_shape = action_shape
        self.dueling = dueling
        if self.dueling:
            self._value_fc = tf.keras.layers.Dense(
                units,
                activation='relu',
                kernel_initializer=tf.random_uniform_initializer(-0.05, 0.05),
                bias_initializer=tf.constant_initializer(0.1)
            )
            self._value_head = tf.keras.layers.Dense(
                1,
                activation=None,
                kernel_initializer=tf.random_uniform_initializer(-0.05, 0.05),
                bias_initializer=tf.constant_initializer(0.1)
            )
            self._adv_fc = tf.keras.layers.Dense(
                units,
                activation='relu',
                kernel_initializer=tf.random_uniform_initializer(-0.05, 0.05),
                bias_initializer=tf.constant_initializer(0.1)
            )
            self._adv_head = tf.keras.layers.Dense(
                action_shape,
                activation=None,
                kernel_initializer=tf.random_uniform_initializer(-0.05, 0.05),
                bias_initializer=tf.constant_initializer(0.1)
            )
        else:
            self._dense = tf.keras.layers.Dense(
                units,
                activation='relu',
                kernel_initializer=tf.random_uniform_initializer(-0.05, 0.05),
                bias_initializer=tf.constant_initializer(0.1)
            )
            self._qvals = tf.keras.layers.Dense(
                action_shape,
                activation=None,
                kernel_initializer=tf.random_uniform_initializer(-0.05, 0.05),
                bias_initializer=tf.constant_initializer(0.1)
            )

    def call(self, inputs, **kwargs):
        if self.dueling:
            svals = self._value_fc(inputs)
            svals = self._value_head(svals)
            advals = self._adv_fc(inputs)
            advals = self._adv_head(advals)
            qvals = svals + (advals - tf.reduce_mean(advals))
            return qvals
        else:
            x = self._dense(inputs)
            return self._qvals(x)
