from abc import ABC, abstractmethod


class Policy(ABC):
    """ Abstract policy class"""

    def __init__(self, state_shape, action_shape):
        self._state_shape = state_shape
        self._action_shape = action_shape

    def act(self, obs, mask=None, training=True):
        return self._act(obs, mask=mask, training=training)

    def distribution(self, obs):
        return self._distribution(obs)

    def get(self, attr):
        if hasattr(self, attr):
            return getattr(self, attr)
        else:
            raise AttributeError(f'Attribute {attr} not found')

    def set(self, attr, val):
        if hasattr(self, attr):
            setattr(self, attr, val)
        else:
            raise AttributeError(f'Attribute {attr} not found')

    @property
    def state_shape(self):
        return self._state_shape

    @property
    def action_shape(self):
        return self._action_shape

    @abstractmethod
    def _act(self, obs, **kwargs):
        pass