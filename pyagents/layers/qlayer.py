import gin
import tensorflow as tf


@gin.configurable
class QLayer(tf.keras.layers.Layer):

    def __init__(self, action_shape, units=32, dropout=None, dueling=True, name='qlayer', **kwargs):
        super().__init__(name=name, **kwargs)
        self.action_shape = action_shape
        self.dueling = dueling
        self.dropout = bool(dropout is not None)
        hidden_initializer = tf.keras.initializers.Orthogonal(0.01)
        final_initializer = tf.keras.initializers.Orthogonal(0.005)
        if self.dueling:
            self._value_fc = tf.keras.layers.Dense(
                units,
                activation='relu',
                # kernel_initializer=hidden_initializer,
                bias_initializer=tf.constant_initializer(0.1)
            )
            if self.dropout:
                self._value_dropout = tf.keras.layers.Dropout(rate=dropout)
            self._value_head = tf.keras.layers.Dense(
                1,
                activation=None,
                # kernel_initializer=final_initializer,
                bias_initializer=tf.constant_initializer(0.1)
            )
            self._adv_fc = tf.keras.layers.Dense(
                units,
                activation='relu',
                # kernel_initializer=hidden_initializer,
                bias_initializer=tf.constant_initializer(0.1)
            )
            if self.dropout:
                self._adv_dropout = tf.keras.layers.Dropout(rate=dropout)
            self._adv_head = tf.keras.layers.Dense(
                action_shape,
                activation=None,
                # kernel_initializer=final_initializer,
                bias_initializer=tf.constant_initializer(0.1)
            )
        else:
            self._dense = tf.keras.layers.Dense(
                units,
                activation='relu',
                kernel_initializer=hidden_initializer,
                bias_initializer=tf.constant_initializer(0.1)
            )
            if self.dropout:
                self._dropout = tf.keras.layers.Dropout(rate=dropout)
            self._qvals = tf.keras.layers.Dense(
                action_shape,
                activation=None,
                kernel_initializer=final_initializer,
                bias_initializer=tf.constant_initializer(0.01)
            )

    def call(self, inputs, training=False, **kwargs):
        if self.dueling:
            svals = self._value_fc(inputs)
            if self.dropout:
                svals = self._value_dropout(svals, training=training)
            svals = self._value_head(svals)
            advals = self._adv_fc(inputs)
            if self.dropout:
                advals = self._adv_dropout(advals, training=training)
            advals = self._adv_head(advals)
            qvals = svals + (advals - tf.reduce_mean(advals))
            return qvals
        else:
            x = self._dense(inputs)
            if self.dropout:
                x = self._dropout(x, training=training)
            return self._qvals(x)
