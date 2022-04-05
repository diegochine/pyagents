import gin
import tensorflow as tf
from pyagents.networks.network import Network
from pyagents.networks.encoding_network import EncodingNetwork
from pyagents.layers import GaussianLayer, DirichletLayer, SoftmaxLayer
from pyagents.policies import GaussianPolicy, DirichletPolicy, SoftmaxPolicy


@gin.configurable
class PolicyNetwork(Network):

    def __init__(self,
                 state_shape,
                 action_shape,
                 distribution='beta',
                 preprocessing_layers=None,
                 conv_layer_params=None,
                 fc_layer_params=(64, 64),
                 dropout_params=None,
                 activation='relu',
                 name='ActorNetwork',
                 trainable=True,
                 dtype=tf.float32
                 ):
        super(PolicyNetwork, self).__init__(name, trainable, dtype)
        self._config = {'state_shape': state_shape,
                        'action_shape': action_shape,
                        'preprocessing_layers': [lay.config() for lay in preprocessing_layers]
                        if preprocessing_layers else [],
                        'conv_layer_params': conv_layer_params if conv_layer_params else [],
                        'fc_layer_params': fc_layer_params if fc_layer_params else [],
                        'dropout_params': dropout_params if dropout_params else [],
                        'activation': activation,
                        'name': name}
        if all(p is None for p in (preprocessing_layers, conv_layer_params, fc_layer_params)):
            self._encoder = None
        else:
            self._encoder = EncodingNetwork(
                state_shape,
                preprocessing_layers=preprocessing_layers,
                conv_layer_params=conv_layer_params,
                fc_layer_params=fc_layer_params,
                dropout_params=dropout_params,
                activation=activation,
            )
        self._out_distribution = distribution
        features_shape = (fc_layer_params[-1], ) if fc_layer_params is not None else state_shape
        if distribution == 'gaussian':
            self._out_layer = GaussianLayer(features_shape, action_shape)
        elif distribution == 'beta':
            self._out_layer = DirichletLayer(features_shape, action_shape)
        elif distribution == 'softmax':
            self._out_layer = SoftmaxLayer(features_shape, action_shape)

    def get_policy(self, caller=None):
        # caller parameter is used when this network is not the main network (eg when used as policy head)
        if self._out_distribution == 'gaussian':
            return GaussianPolicy(self._config['state_shape'], self._config['action_shape'],
                                  caller if caller is not None else self)
        elif self._out_distribution == 'beta':
            return DirichletPolicy(self._config['state_shape'], self._config['action_shape'],
                                   caller if caller is not None else self, bounds=(-2, 2)) # TODO improve this hardcoded bounds
        elif self._out_distribution == 'softmax':
            return SoftmaxPolicy(self._config['state_shape'], self._config['action_shape'],
                                 caller if caller is not None else self)

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

