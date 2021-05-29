import gin
import tensorflow as tf

from networks.network import Network
from networks.encoding_network import EncodingNetwork


@gin.configurable
class QNetwork(Network):

    def __init__(self, state_shape, action_shape, preprocessing_layers=None,
                 conv_layer_params=None, fc_layer_params=(128, 64), activation=tf.keras.activations.relu,
                 name='QNetwork'):
        super().__init__(name=name)
        self._encoder = EncodingNetwork(
            state_shape,
            preprocessing_layers=preprocessing_layers,
            conv_layer_params=conv_layer_params,
            fc_layer_params=fc_layer_params,
            activation=activation
        )
        self._q_layer = tf.keras.layers.Dense(
            action_shape,
            activation=None,
            kernel_initializer=tf.random_uniform_initializer(-0.05, 0.05),
            bias_initializer=tf.constant_initializer(-0.1)
        )

    def call(self, inputs, training=False, mask=None):
        state = self._encoder(inputs)
        q_values = self._q_layer(state)
        return q_values

    def get_config(self):
        pass
