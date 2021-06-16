import gin
import tensorflow as tf

from pyagents.networks.network import Network
from pyagents.networks.encoding_network import EncodingNetwork


@gin.configurable
class QNetwork(Network):

    def __init__(self, state_shape, action_shape, preprocessing_layers=None,
                 conv_layer_params=None, fc_layer_params=(128, 64), activation='relu',
                 name='QNetwork', trainable=True, dtype=tf.float32):
        super().__init__(name=name, trainable=trainable, dtype=dtype)
        self._config = {'state_shape': state_shape,
                        'action_shape': action_shape,
                        'preprocessing_layers': [lay.get_config() for lay in preprocessing_layers]
                        if preprocessing_layers else [],
                        'conv_layer_params': conv_layer_params if conv_layer_params else [],
                        'fc_layer_params': fc_layer_params if fc_layer_params else [],
                        'activation': activation,
                        'name': name}
        self._encoder = EncodingNetwork(
            state_shape,
            preprocessing_layers=preprocessing_layers,
            conv_layer_params=conv_layer_params,
            fc_layer_params=fc_layer_params,
            activation=activation,
            name=name
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
        config = super(QNetwork, self).get_config()
        config.update(self._config)
        return config

    @classmethod
    def from_config(cls, config):
        """Be careful with the type of the preprocessing layers:
        they must be a list of already built layer and not a list
        of their configurations."""
        return cls(**config)
