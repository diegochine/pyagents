import abc
from abc import ABC

import keras
import tensorflow as tf


class Network(tf.keras.layers.Layer, ABC):

    def __init__(self, name):
        super(Network, self).__init__(name=name)

    @abc.abstractmethod
    def call(self, inputs, training=None, mask=None):
        pass
