from typing import Optional, Iterable, Union

import gin
import numpy as np
import tensorflow as tf
from pyagents.networks.network import Network, NetworkOutput
from pyagents.networks.encoding_network import EncodingNetwork
from pyagents.layers import GaussianLayer, DirichletLayer, SoftmaxLayer
from pyagents.policies import GaussianPolicy, DirichletPolicy, SoftmaxPolicy, FixedPolicy


@gin.configurable
class PolicyNetwork(Network):
    """A Network representing pi(a | s, Theta)"""

    def __init__(self,
                 state_shape: tuple,
                 action_shape: tuple,
                 output: str,
                 conv_params: Optional[Iterable] = None,
                 fc_params: Optional[Iterable[int]] = (64, 64),
                 dropout_params: Optional[Union[float, Iterable[float]]] = None,
                 activation: str = 'tanh',
                 out_params: Optional[dict] = None,
                 bounds: Optional[tuple] = None,
                 name: str = 'PolicyNetwork',
                 trainable: bool = True,
                 dtype=tf.float32):
        """Creates a Policy Network.

        Args:
            state_shape: Tuple representing the shape of the input space (observations).
            action_shape: Tuple representing the shape of the action space.
            output: Output distribution of network, either 'continuous' (no distribution parametrization),
              'gaussian', 'dirichlet' or 'softmax'.
            conv_params: (Optional) Iterable of hyperparameters for hidden convolutional layers in encoder network.
              Defaults to None (no conv used).
            fc_params: (Optional) Number of units for each hidden fully connected layer in encoder network.
              Defaults to (64, 64).
            dropout_params: (Optional) Dropout hyperparameters for encoder network. Defaults to None (no dropout).
            activation: (Optional) Name of activation function of encoder network. Defaults to 'relu'.
            out_params: (Optional) Additional parameters for output layer. Defaults to None.
            bounds: (Optional) Bounds of action space. Default to None (unbounded space).
            name:  Name of the network. Defaults to 'ActorNetwork'.
            trainable: if True, network is trainable. Defaults to True.
            dtype: Network dtype. Defaults to tf.float32.
        """
        super(PolicyNetwork, self).__init__(name, trainable, dtype)
        self._config = {'state_shape': state_shape,
                        'action_shape': action_shape,
                        'conv_params': conv_params if conv_params else [],
                        'fc_params': fc_params if fc_params else [],
                        'dropout_params': dropout_params if dropout_params else [],
                        'bounds': bounds if bounds else [],
                        'activation': activation,
                        'output': output,
                        'out_params': out_params,
                        'name': name}
        if out_params is None:
            out_params = {}
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
        assert bounds is None or len(bounds) == 2, f'wrong bounds param: {bounds}'
        if bounds is not None:
            if not isinstance(bounds, np.ndarray):
                bounds = np.array(bounds, dtype=dtype.as_numpy_dtype if isinstance(dtype, tf.DType) else dtype)
            bounds = (tf.convert_to_tensor(bounds[0].squeeze()), tf.convert_to_tensor(bounds[1].squeeze()))
        self._bounds = bounds
        self._output_type = output
        features_shape = (fc_params[-1],) if fc_params is not None else state_shape  # input shape for out
        if output == 'continuous':
            self._out_layer = tf.keras.layers.Dense(action_shape[0], **out_params,
                                                    kernel_initializer=tf.keras.initializers.Orthogonal(0.01))
        elif output == 'gaussian':
            self._out_layer = GaussianLayer(features_shape, action_shape, bounds=bounds, **out_params)
        elif output == 'beta':
            self._out_layer = DirichletLayer(features_shape, action_shape, bounds=bounds, **out_params)
        elif output == 'softmax':
            self._out_layer = SoftmaxLayer(features_shape, action_shape, **out_params)
        else:
            raise ValueError(f'unknown output type {output}')

    def get_policy(self, caller=None):
        """Returns a Policy object that represents the this network's current policy."""
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

    def call(self, inputs, training=True, mask=None) -> NetworkOutput:
        if self._encoder is not None:
            state = self._encoder(inputs, training=training)
        else:
            state = inputs
        output = self._out_layer(state, training=training)
        if self._output_type in ('gaussian', 'beta', 'softmax'):
            action, dist_params, logprobs = output
        else:  # deterministic policy network
            action = output
            dist_params = None
            logprobs = None
        # if self._bounds is not None:  # FIXME secondo me da spostare su agent
        #     lb, ub = self._bounds
        #     action = tf.clip_by_value(action, lb, ub)
        return NetworkOutput(actions=action,
                             dist_params=dist_params,
                             logprobs=logprobs)
