import random
from collections import deque
import pickle
import gin
from memory.buffer import Buffer


def load_memories(path='./memories/'):
    try:
        with open(path + 'dataset.pkl', 'rb') as f:
            memories = pickle.load(f)
        return memories
    except FileNotFoundError:
        return None


@gin.configurable
class UniformBuffer(Buffer):

    def __init__(self, size_short=5000, size_long=50000, ltmemory=None, save_dir='./memories'):
        super().__init__(save_dir)
        if ltmemory is not None:
            self.ltmemory = ltmemory
        else:
            self.ltmemory = deque(maxlen=size_long)
        self.stmemory = deque(maxlen=size_short)

    def __len__(self):
        return len(self.ltmemory)

    def commit_stmemory(self, fragment, data_augmentation=None):
        """
        :param fragment: dictionary to save
        :param data_augmentation: function to be applied in order to do data augmentation
        """
        if data_augmentation is not None:
            pass

        self.stmemory.append(fragment)

    def commit_ltmemory(self, **kwargs):
        for mem in self.stmemory:
            self.ltmemory.append(mem)
        self.clear_stmemory()

    def clear_stmemory(self):
        self.stmemory.clear()

    def sample(self, batch_size, vectorizing_fn=lambda x: x):
        return vectorizing_fn(random.sample(list(self.ltmemory), batch_size)), []  # no need to return samples indexes

    def save(self, name):
        pickle.dump(self.ltmemory, open(f'{self._save_dir}/mem-{name}.pkl', 'wb'))

    def clear_ltmemory(self):
        self.ltmemory.clear()