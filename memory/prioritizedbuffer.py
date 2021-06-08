import numpy as np
import gin
from collections import deque
from memory.buffer import Buffer


# TODO implement importance sampling (see https://danieltakeshi.github.io/2019/07/14/per/)
@gin.configurable
class PrioritizedBuffer(Buffer):

    def __init__(self, save_dir=None, size_short=5000, size_long=50000, ltmemory=None, eps=0.02, alpha=0.5, beta=0.4):
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
        self._alpha = alpha
        self._beta = beta

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
        priorities_pow = np.power(priorities, self._alpha)
        probs = priorities_pow / priorities_pow.sum()
        indexes = np.random.choice(list(self.ltmemory.keys()), size=batch_size, replace=False, p=probs)
        samples = [self.ltmemory[idx][0] for idx in indexes]
        is_weights = np.power(1/(len(self.ltmemory) * probs[indexes]), self._beta)
        is_weights = is_weights / np.max(is_weights)
        return vectorizing_fn(samples), indexes, is_weights

    def clear_ltmemory(self):
        self.ltmemory.clear()

    def update_samples(self, errors, indexes):
        assert errors.shape == indexes.shape
        for error_idx, mem_idx in enumerate(indexes):
            self.ltmemory[mem_idx][1] = errors[error_idx].numpy()
