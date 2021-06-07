import numpy as np
import gin
from collections import deque
from memory.buffer import Buffer


# TODO implement importance sampling (see https://danieltakeshi.github.io/2019/07/14/per/)
@gin.configurable
class PrioritizedBuffer(Buffer):

    def __init__(self, save_dir, size_short=5000, size_long=50000, ltmemory=None, eps=0.02):
        super().__init__(save_dir)
        if ltmemory is not None:
            self.ltmemory = ltmemory
        else:
            self.ltmemory = {}
        self._size_long = size_long
        self.stmemory = deque(maxlen=size_short)
        self._idx = 0
        self._max_p = 1/size_long
        self._eps = eps

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
            self.ltmemory[self._idx] = [mem, self._max_p]
            self._idx = (self._idx + 1) % self._size_long
        self.clear_stmemory()

    def clear_stmemory(self):
        self.stmemory.clear()

    def sample(self, batch_size, vectorizing_fn=lambda x: x):
        priorities = np.array([self.ltmemory[idx][1] for idx in self.ltmemory]) + self._eps
        probs = priorities / np.sum(priorities)
        indexes = np.random.choice(list(self.ltmemory.keys()), size=batch_size, replace=False, p=probs)
        return vectorizing_fn([self.ltmemory[idx][0] for idx in indexes]), indexes

    def clear_ltmemory(self):
        self.ltmemory.clear()

    def update_samples(self, errors, indexes):
        assert errors.shape == indexes.shape
        for error_idx, mem_idx in enumerate(indexes):
            self.ltmemory[mem_idx][1] = errors[error_idx].numpy()
