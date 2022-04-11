import gin
import numpy as np
import tensorflow as tf
from pyagents.networks.network import Network
from pyagents.networks.encoding_network import EncodingNetwork
from pyagents.layers import GaussianLayer, DirichletLayer, SoftmaxLayer
from pyagents.policies import GaussianPolicy, DirichletPolicy, SoftmaxPolicy, FixedPolicy


@gin.configurable
class PolicyNetwork(Network):

    def __init__(self,
                 state_shape,
                 action_shape,
                 output='gaussian',
                 conv_params=None,
                 fc_params=(64, 64),
                 dropout_params=None,
                 activation='relu',
                 out_params=None,
                 bounds=None,
                 action_noise=0.1,
                 name='ActorNetwork',
                 trainable=True,
                 dtype=tf.float32):
        super(PolicyNetwork, self).__init__(name, trainable, dtype)
        if out_params is None:
            out_params = {}
        self._config = {'state_shape': state_shape,
                        'action_shape': action_shape,
                        'conv_params': conv_params if conv_params else [],
                        'fc_params': fc_params if fc_params else [],
                        'dropout_params': dropout_params if dropout_params else [],
                        'activation': activation,
                        'action_noise': action_noise,
                        'output': output,
                        'out_params': out_params,
                        'name': name}
        if conv_params is None and fc_params is None:
            self._encoder = None
        else:
            self._encoder = EncodingNetwork(
                state_shape,
                conv_params=conv_params,
                fc_params=fc_params,
                dropout_params=dropout_params,
                activation=activation,
            )
        self._bounds = bounds if bounds is not None else (-np.inf, np.inf)
        self._output_type = output
        features_shape = (fc_params[-1],) if fc_params is not None else state_shape  # input shape for out
        if output == 'continuous':
            self._out_layer = tf.keras.layers.Dense(action_shape[0], **out_params,
                                                    kernel_initializer=tf.random_uniform_initializer(minval=-0.003, maxval=0.003))
        elif output == 'gaussian':
            self._out_layer = GaussianLayer(features_shape, action_shape)
        elif output == 'beta':
            self._out_layer = DirichletLayer(features_shape, action_shape)
        elif output == 'softmax':
            self._out_layer = SoftmaxLayer(features_shape, action_shape)

    def get_policy(self, caller=None):
        # caller parameter is used when this network is not the main network (eg when used as policy head)
        if self._output_type == 'continuous':
            return FixedPolicy(state_shape=self._config['state_shape'],
                               action_shape=self._config['action_shape'],
                               policy_network=caller if caller is not None else self,
                               bounds=self._bounds)
        elif self._output_type == 'gaussian':
            return GaussianPolicy(state_shape=self._config['state_shape'],
                                  action_shape=self._config['action_shape'],
                                  policy_network=caller if caller is not None else self,
                                  bounds=self._bounds)
        elif self._output_type == 'beta':
            return DirichletPolicy(state_shape=self._config['state_shape'],
                                   action_shape=self._config['action_shape'],
                                   policy_network=caller if caller is not None else self,
                                   bounds=self._bounds)
        elif self._output_type == 'softmax':
            return SoftmaxPolicy(state_shape=self._config['state_shape'],
                                 action_shape=self._config['action_shape'],
                                 policy_network=caller if caller is not None else self)

    def get_config(self):
        config = super().get_config()
        config.update(self._config)
        return config

    def call(self, inputs, training=True, mask=None):
        if self._encoder is not None:
            state = self._encoder(inputs, training=training)
        else:
            state = inputs
        dist_params = self._out_layer(state, training=training)
        return dist_params
