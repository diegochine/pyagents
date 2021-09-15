import os
import pickle
import gin
from abc import ABC, abstractmethod


def load_memories(path):
    try:
        with open(os.path.join(path, 'memories.pkl'), 'rb') as f:
            memories = pickle.load(f)
        return memories
    except FileNotFoundError:
        return None


@gin.configurable
class Buffer(ABC):
    """ Abstract class for all buffers"""

    def __init__(self, save_dir=None):
        if save_dir is not None and not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        self._save_dir = save_dir

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def sample(self, batch_size, vectorizing_fn):
        pass

    @abstractmethod
    def get_config(self):
        pass

    def set_save_dir(self, path):
        self._save_dir = path

    def save(self):
        if self._save_dir is None:
            raise ValueError('Buffer needs a save_dir path')
        pickle.dump(self, open(os.path.join(self._save_dir, 'memories.pkl'), 'wb'))

    def update_samples(self, errors, indexes):
        """ subclasses may optionally implement this method"""
        pass
