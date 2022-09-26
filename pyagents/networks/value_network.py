from typing import Union

import gin
import numpy as np
import tensorflow as tf
from pyagents.networks.network import Network, NetworkOutput
from pyagents.networks.encoding_network import EncodingNetwork


@gin.configurable
class ValueNetwork(Network):

    def __init__(self,
                 state_shape,
                 conv_params=None,
                 fc_params=(64, 64),
                 dropout_params=None,
                 activation: str = 'tanh',
                 init: bool = True,
                 name: str = 'ValueNetwork',
                 trainable: bool = True,
                 dtype: str = 'float32'):
        super(ValueNetwork, self).__init__(name, trainable, dtype)
        self._config = {'state_shape': state_shape,
                        'conv_params': conv_params if conv_params else [],
                        'fc_params': fc_params if fc_params else [],
                        'dropout_params': dropout_params if dropout_params else [],
                        'activation': activation,
                        'name': name,
                        'init': init,
                        'dtype': dtype}
        if conv_params is None and fc_params is None:
            self._encoder = None
        else:
            self._encoder = EncodingNetwork(
                state_shape,
                conv_params=conv_params,
                fc_params=fc_params,
                dropout_params=dropout_params,
                activation=activation,
                dtype=dtype
            )
        self._value_head = tf.keras.layers.Dense(1,
                                                 kernel_initializer=tf.keras.initializers.Orthogonal(1.0),
                                                 dtype=dtype)
        self._denormalizer = None
        if init:
            self(tf.ones((1, *state_shape)))

    def get_config(self):
        config = super().get_config()
        config.update(self._config)
        return config

    def set_denormalizer(self, mean: Union[float, np.array], std: Union[float, np.array]):
        if not isinstance(mean, tf.Tensor):
            mean = tf.convert_to_tensor(mean, dtype=self.dtype)
        if not isinstance(std, tf.Tensor):
            std = tf.convert_to_tensor(std, dtype=self.dtype)
        assert mean.dtype == self.dtype and std.dtype == self.dtype, 'provided normalizer tensors with wrong dtype'
        self._denormalizer = (mean, std)

    def call(self, inputs, training=True, denormalize=True, mask=None):
        if self._encoder is not None:
            state = self._encoder(inputs, training=training)
        else:
            state = inputs
        value = self._value_head(state, training=training)
        if self._denormalizer is not None and denormalize:
            mean, std = self._denormalizer
            value = mean + (value * tf.maximum(std, 1e-6))
        return NetworkOutput(critic_values=value)

