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

    def __init__(self, state_shape, action_shape, name=None):
        super(Agent, self).__init__(name=name)
        self._state_shape = state_shape
        self._action_shape = action_shape

    def _initialize(self):
        pass

    def initialize(self):
        pass

    @abc.abstractmethod
    def act(self, state):
        """Returns the best action in the state according to current policy"""

    def _loss(self, memories):
        """Computes loss"""
        raise NotImplementedError("Loss not implemented.")

    @abc.abstractmethod
    def _train(self, batch_size):
        pass

    def train(self, batch_size):
        return self._train(batch_size)
