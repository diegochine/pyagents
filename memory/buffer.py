import os
import pickle
import gin
from abc import ABC, abstractmethod


def load_memories(path='./memories/'):
    try:
        with open(path + 'dataset.pkl', 'rb') as f:
            memories = pickle.load(f)
        return memories
    except FileNotFoundError:
        return None


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

    def save(self, name):
        pickle.dump(self.ltmemory, open(f'{self._save_dir}/mem-{name}.pkl', 'wb'))

    def update_samples(self, errors, indexes):
        """ subclasses may optionally implement this method"""
        pass
