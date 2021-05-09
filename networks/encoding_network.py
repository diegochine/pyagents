import gin
import tensorflow as tf
from keras import Input
from keras.layers import Dense, Conv1D, Conv2D
from keras.activations import relu
from keras.regularizers import l2, l1_l2
from keras.optimizers import Adam
from keras.models import Model
from networks import Network


@gin.configurable
class EncodingNetwork(Network):

    def __init__(self, state_shape,
                 conv_layer_params=None, fc_layer_params=None, activation=relu,
                 name='EncodingNetwork'):
        super().__init__(name=name)
        self.model = self._build_model()

    def _build_model(self):
        inputs = Input(shape=self.state_shape)
        x = Dense(16, activation='relu',
                  kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                  bias_regularizer=l2(1e-4))(inputs)
        x = Dense(32, activation='relu',
                  kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                  bias_regularizer=l2(1e-4))(x)
        x = Dense(16, activation='relu',
                  kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                  bias_regularizer=l2(1e-4))(x)
        # output layer, linear because the output are qvals
        outputs = Dense(self.action_shape, activation='linear')(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='mse', optimizer=Adam(self.learning_rate))

        return model

    def call(self, inputs, training=False, mask=None):
        return self.model(inputs)

    def get_config(self):
        pass
