import random
import gin
from collections import deque

import numpy as np

from pydrlagents.memory.buffer import Buffer


@gin.configurable
class UniformBuffer(Buffer):

    def __init__(self, save_dir=None, size_short=5000, size_long=50000, ltmemory=None):
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
        # no need to return samples indexes, and is_weights contains all ones (as it's not used)
        return vectorizing_fn(random.sample(list(self.ltmemory), batch_size)), [], np.ones(batch_size)

    def clear_ltmemory(self):
        self.ltmemory.clear()
