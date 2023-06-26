import gin
import numpy as np
import torch
import torch.nn as nn
from pyagents.networks.network import Network
from pyagents.layers.noisynet import NoisyLayer


@gin.configurable
class EncodingNetwork(Network):

    def __init__(self,
                 state_shape,
                 conv_params=None,
                 fc_params=None,
                 dropout_params=None,
                 activation='tanh',
                 noisy_layers=False,
                 dtype: str = 'float32',
                 do_init: bool = True,
                 name='EncodingNetwork',
                 conv_type='2d'):
        super().__init__(name=name, dtype=dtype)
        self._state_shape = state_shape
        self.noisy_layers = noisy_layers
        self._config = {'state_shape': state_shape,
                        'conv_params': conv_params if conv_params else tuple(),
                        'fc_params': fc_params if conv_params else tuple(),
                        'dropout_params': dropout_params if dropout_params else tuple(),
                        'activation': activation,
                        'name': name,
                        'do_init': do_init,
                        'noisy_layers': noisy_layers,
                        'dtype': dtype,
                        'conv_type': conv_type}

        # TODO improve inizialization, allow initizializer to be passed as parameter
        kernel_initializer = tf.keras.initializers.Orthogonal(0.1)

        layers = []

        if conv_params:
            if conv_type == '2d':
                conv_layer_type = nn.Conv2d
            elif conv_type == '1d':
                conv_layer_type = nn.Conv1d
            else:
                raise ValueError(f'unsupported conv type {conv_type}. Use "1d" or "2d"')

            for config in conv_params:
                if len(config) == 4:
                    (filters, kernel_size, strides, dilation_rate) = config
                elif len(config) == 3:
                    (filters, kernel_size, strides) = config
                    dilation_rate = (1, 1) if conv_type == '2d' else (1,)
                else:
                    raise ValueError('only 3 or 4 elements permitted in conv_params tuples')
                layers.append(conv_layer_type(out_channels=filters,
                                              kernel_size=kernel_size,
                                              stride=strides,
                                              dilation=dilation_rate,
                                              activation=activation,
                                              dtype=dtype))

        layers.append(tf.keras.layers.Flatten())
        if not dropout_params:
            dropout_params = None
        if dropout_params is None or isinstance(dropout_params, float):
            dropout_params = [dropout_params] * len(fc_params)
        else:
            assert isinstance(dropout_params,
                              (tuple, list)), f'unrecognized type for dropout_params {type(dropout_params)} '
        if fc_params:
            if noisy_layers:
                fc_layer = NoisyLayer
            else:
                fc_layer = tf.keras.layers.Dense
            assert len(fc_params) == len(
                dropout_params), f'params length do not match (fc: {len(fc_params)}, dropout: {len(dropout_params)})'
            for num_units, dropout in zip(fc_params, dropout_params):
                kernel_regularizer = None  # if necessary sholud have wheight decay param as in tf
                layers.append(fc_layer(
                    units=num_units,
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer,
                    dtype=dtype))
                if dropout is not None:
                    layers.append(tf.keras.layers.Dropout(rate=dropout))

        self._postprocessing_layers = layers
        if do_init:
            self(torch.ones((1, *state_shape)))

    def forward(self, inputs, training=False, mask=None):
        states = inputs
        for layer in self._postprocessing_layers:
            states = layer(states, training=training)
        return states

    def get_config(self):
        config = super(EncodingNetwork, self).get_config()
        config.update(self._config)
        return config

    @classmethod
    def from_config(cls, config):
        """Be careful with the type of the preprocessing layers:
        they must be a list of already built layer and not a list
        of their configurations."""
        return cls(**config)

    def reset_noise(self):
        """Resets noise of noisy layers"""
        for layer in self._postprocessing_layers:
            if isinstance(layer, NoisyLayer):
                layer.reset_noise()
