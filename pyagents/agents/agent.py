import abc
import tensorflow as tf


class Agent(tf.Module, abc.ABC):
    """Abstract base class for all implemented agents.
    The agent serves two purposes:
    * Training by reading minibatches of memories, and updating an internal
      neural network (train method)
    * TODO expose policy to interact with environment

    Main methods:
    * initialize: perform initialization before training
    * train: read minibatch of memories from a replay buffer and updates
      some internal network
    """

    def __init__(self, state_shape, action_shape, training, name='Agent'):
        super(Agent, self).__init__(name=name)
        self._state_shape = state_shape
        self._action_shape = action_shape
        self._training = training

    @property
    def state_shape(self):
        return self._state_shape

    @property
    def action_shape(self):
        return self._action_shape

    @property
    def policy(self):
        return self._policy

    def toggle_training(self, training=None):
        self._training = not self._training if training is None else training

    def _initialize(self):
        pass

    def initialize(self):
        pass

    def act(self, state, mask=None):
        """Returns the best action in the state according to current policy"""
        return self._policy.act(state, mask=mask, training=self._training)

    def _loss(self, memories):
        """Computes loss"""
        raise NotImplementedError("Loss not implemented.")

    @abc.abstractmethod
    def _train(self, batch_size):
        pass

    def train(self, batch_size=None):
        return self._train(batch_size)

    @abc.abstractmethod
    def save(self, path):
        pass

    @classmethod
    @abc.abstractmethod
    def load(cls, path, ver):
        pass
