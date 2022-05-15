from abc import ABC, abstractmethod
from collections import namedtuple

import tensorflow as tf

NetworkOutput = namedtuple('NetworkOutput',
                           ('actions', 'dist_params', 'critic_values', 'logprobs', 'logits'),
                           defaults=(None, None, None, None, None))  # no parameter required


class Network(tf.keras.layers.Layer, ABC):

    def __init__(self, name, trainable=True, dtype=tf.float32):
        super(Network, self).__init__(name=name, trainable=trainable, dtype=dtype)

    @abstractmethod
    def call(self, inputs, training=None, mask=None) -> NetworkOutput:
        pass

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @property
    def variables(self):
        if not self.built:
            raise ValueError("Network has not been built, unable to access variables.")
        return super(Network, self).variables

    @property
    def trainable_variables(self):
        if not self.built:
            raise ValueError("Network has not been built, unable to access variables.")
        return super(Network, self).trainable_variables
