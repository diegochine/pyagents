import gin
import tensorflow as tf

from pyagents.layers.noisynet import NoisyLayer


@gin.configurable
class QLayer(tf.keras.layers.Layer):

    def __init__(self,
                 action_shape,
                 units=32,
                 dropout=None,
                 dueling=True,
                 noisy_layers=False,
                 n=None,
                 name='qlayer',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        if n is not None:  # either number of atoms (c51) or number of quantiles (qr dqn)
            # assert isinstance(n, int), f'distributional q layer need n to be int'
            self.distributional = True
        else:
            self.distributional = False
            n = 1
        self.n = n
        if noisy_layers:
            fc_layer = NoisyLayer
        else:
            fc_layer = tf.keras.layers.Dense
        self.noisy_layers = noisy_layers
        self.action_shape = action_shape
        self.dueling = dueling
        self.dropout = bool(dropout is not None)
        # hidden_initializer = tf.random_uniform_initializer(-0.1, 0.1)
        # final_initializer = tf.random_uniform_initializer(-0.005, 0.005)
        if self.dueling:
            self._value_fc = fc_layer(
                units,
                activation='relu',
                # kernel_initializer=hidden_initializer,
                # bias_initializer=tf.constant_initializer(0.1)
            )
            if self.dropout:
                self._value_dropout = tf.keras.layers.Dropout(rate=dropout)
            self._value_head = fc_layer(
                n,
                activation=None,
                # kernel_initializer=final_initializer,
                # bias_initializer=tf.constant_initializer(0.1)
            )
            self._adv_fc = fc_layer(
                units,
                activation='relu',
                # kernel_initializer=hidden_initializer,
                # bias_initializer=tf.constant_initializer(0.1)
            )
            if self.dropout:
                self._adv_dropout = tf.keras.layers.Dropout(rate=dropout)
            self._adv_head = fc_layer(
                n * action_shape,
                activation=None,
                # kernel_initializer=final_initializer,
                # bias_initializer=tf.constant_initializer(0.1)
            )
        else:
            self._dense = fc_layer(
                units,
                activation='relu',
                # kernel_initializer=hidden_initializer,
                # bias_initializer=tf.constant_initializer(0.1)
            )
            if self.dropout:
                self._dropout = tf.keras.layers.Dropout(rate=dropout)
            self._qvals = fc_layer(
                n * action_shape,
                activation=None,
                # kernel_initializer=final_initializer,
                # bias_initializer=tf.constant_initializer(0.01)
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
            if self.distributional:
                svals = tf.reshape(svals, (-1, 1, self.n))
                advals = tf.reshape(advals, (-1, self.action_shape, self.n))
            qvals = svals + (advals - tf.reduce_mean(advals, axis=1, keepdims=True))
            return qvals
        else:
            x = self._dense(inputs)
            if self.dropout:
                x = self._dropout(x, training=training)
            qvals = self._qvals(x)
            if self.distributional:
                qvals = tf.reshape(qvals, (-1, self.action_shape, self.n))
            return qvals

    def reset_noise(self):
        """Resets noise of noisy layers"""
        if self.noisy_layers:
            if self.dueling:
                self._adv_fc.reset_noise()
                self._value_fc.reset_noise()
            else:
                self._dense.reset_noise()
