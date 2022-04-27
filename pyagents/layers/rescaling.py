import tensorflow as tf


class RescalingLayer(tf.keras.layers.Layer):
    """A simple layer to perform arbitrary rescaling of values."""

    def __init__(self, scaling_factor=1.0, name='Rescaling', dtype=tf.float32, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self._scaling_factor = scaling_factor

    def get_config(self):
        return {'scaling_factor': self._scaling_factor}

    def call(self, inputs):
        return self._scaling_factor * inputs
