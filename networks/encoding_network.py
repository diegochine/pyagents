import gin
import tensorflow as tf
from networks.network import Network


def _copy_layer(layer):
    """Create a copy of a Keras layer with identical parameters.
  The new layer will not share weights with the old one.
  Args:
    layer: An instance of `tf.keras.layers.Layer`.
  Returns:
    A new keras layer.
  Raises:
    TypeError: If `layer` is not a keras layer.
    ValueError: If `layer` cannot be correctly cloned.
  """
    if not isinstance(layer, tf.keras.layers.Layer):
        raise TypeError('layer is not a keras layer: %s' % str(layer))

    if layer.built:
        logging.warning(
            'Beware: Copying a layer that has already been built: \'%s\'.  '
            'This can lead to subtle bugs because the original layer\'s weights '
            'will not be used in the copy.', layer.name)
    # Get a fresh copy so we don't modify an incoming layer in place.  Weights
    # will not be shared.
    return type(layer).from_config(layer.get_config())


@gin.configurable
class EncodingNetwork(Network):

    def __init__(self, state_shape, preprocessing_layers=None,
                 conv_layer_params=None, fc_layer_params=None, activation='relu', dtype=tf.float32,
                 name='EncodingNetwork', conv_type='2d'):
        super().__init__(name=name)
        self._state_shape = state_shape
        self._preprocessing_layers = preprocessing_layers

        kernel_initializer = tf.keras.initializers.VarianceScaling(
            scale=2.0, mode='fan_in', distribution='truncated_normal')

        layers = []

        if conv_layer_params:
            if conv_type == '2d':
                conv_layer_type = tf.keras.layers.Conv2D
            elif conv_type == '1d':
                conv_layer_type = tf.keras.layers.Conv1D
            else:
                raise ValueError('unsupported conv type of %s. Use 1d or 2d' % (
                    conv_type))

            for config in conv_layer_params:
                if len(config) == 4:
                    (filters, kernel_size, strides, dilation_rate) = config
                elif len(config) == 3:
                    (filters, kernel_size, strides) = config
                    dilation_rate = (1, 1) if conv_type == '2d' else (1,)
                else:
                    raise ValueError(
                        'only 3 or 4 elements permitted in conv_layer_params tuples')
                layers.append(
                    conv_layer_type(
                        filters=filters,
                        kernel_size=kernel_size,
                        strides=strides,
                        dilation_rate=dilation_rate,
                        activation=activation,
                        kernel_initializer=kernel_initializer,
                        dtype=dtype))

        layers.append(tf.keras.layers.Flatten())

        if fc_layer_params:
            for num_units in fc_layer_params:
                kernal_regularizer = None  # if necessary sholud have wheight decay param as in tf
                layers.append(
                    tf.keras.layers.Dense(
                        num_units,
                        activation=activation,
                        kernel_initializer=kernel_initializer,
                        kernel_regularizer=kernal_regularizer,
                        dtype=dtype))

        # Pull out the nest structure of the preprocessing layers. This avoids
        # saving the original kwarg layers as a class attribute which Keras would
        # then track.
        self._postprocessing_layers = layers
        self.built = True  # Allow access to self.variables
        self._config = {'state_shape': state_shape,
                        'preprocessing_layers': [lay.get_config() for lay in self._preprocessing_layers],
                        'conv_layer_params': conv_layer_params,
                        'fc_layer_params': fc_layer_params,
                        'activation': activation,
                        'dtype': dtype,
                        'name': name,
                        'conv_type': conv_type}

    def call(self, inputs, training=False, mask=None):
        states = inputs
        if self._preprocessing_layers is not None:
            for prep_layer in self._preprocessing_layers:
                states = prep_layer(states, training=training)
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
