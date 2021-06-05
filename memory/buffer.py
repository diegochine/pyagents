import os
from abc import ABC, abstractmethod

import gin


@gin.configurable
class Buffer(ABC):
    """ Abstract class for all buffers"""
    def __init__(self, save_dir='./memories'):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        self._save_dir = save_dir

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def sample(self, batch_size, vectorizing_fn):
        pass

    @abstractmethod
    def save(self, name):
        pass

    def update_samples(self, errors, indexes):
        """ subclasses may optionally implement this method"""
        pass
