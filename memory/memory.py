import os
import random
from collections import deque
import pickle

import gin


def load_memories(path='./memories/'):
    try:
        with open(path + 'dataset.pkl', 'rb') as f:
            memories = pickle.load(f)
        return memories
    except FileNotFoundError:
        return None


def compact_memories(path='./memories/'):
    memories = pickle.load(open(path + 'dataset.pkl', 'rb'))
    for fname in os.listdir(path):
        if fname.endswith('pkl') and not fname.startswith('dataset'):
            with open(path + fname, 'rb') as f:
                memories.extend(pickle.load(f))
            os.remove(path + fname)
    pickle.dump(memories, open(path + 'dataset.pkl', 'wb'))


@gin.configurable
class Memory:

    def __init__(self, size_short=5000, size_long=10000, ltmemory=None, save_dir='./memories'):
        if ltmemory is not None:
            self.ltmemory = ltmemory
        else:
            self.ltmemory = deque(maxlen=size_long)
        self.stmemory = deque(maxlen=size_short)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        self._save_dir = save_dir

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

    def commit_ltmemory(self):
        for mem in self.stmemory:
            self.ltmemory.append(mem)
        self.clear_stmemory()

    def clear_stmemory(self):
        self.stmemory.clear()

    def sample(self, batch_size, vectorizing_fn=lambda x: x):
        return vectorizing_fn(random.sample(list(self.ltmemory), batch_size))

    def save(self, name):
        pickle.dump(self.ltmemory, open(f'{self._save_dir}/mem-{name}.pkl', 'wb'))

    def clear_ltmemory(self):
        self.ltmemory.clear()

    def last_episode(self):
        return self.stmemory
