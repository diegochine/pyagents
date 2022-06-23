import numpy as np
import gin

from pyagents.memory.uniformbuffer import UniformBuffer
from pyagents.memory.sum_tree import SumTree


@gin.configurable
class PrioritizedBuffer(UniformBuffer):

    def __init__(self,
                 save_dir=None,
                 size=100000,
                 n_step_return=1,
                 eps=0.02,
                 alpha=0.6,
                 beta=(0.0, 1.0, 10 ** 6)):
        super().__init__(save_dir=save_dir, n_step_return=n_step_return)
        self._sum_tree = SumTree(size)
        self._size = size
        self._ltmemory = dict()
        self._ptr = 0
        self._eps = eps
        self._alpha = alpha
        if isinstance(beta, int):
            self._beta = beta
            self._beta_max = beta
            self._beta_inc = 0
        elif isinstance(beta, tuple):
            self._beta, self._beta_max, steps = beta
            self._beta_inc = (self._beta_max - self._beta) / steps
        self._config.update({'type': 'PER', 'eps_buffer': eps, 'alpha': alpha,
                             'beta': self._beta, 'beta_max': self._beta_max, 'beta_inc': self._beta_inc})

    def __len__(self):
        return len(self._sum_tree)

    @property
    def n_step_return(self):
        return self._n + 1

    def get_config(self):
        return self._config

    def commit_ltmemory(self, experience):
        self._ltmemory[self._ptr] = experience
        self._sum_tree.set(self._ptr)  # mark as new experience in sum tree
        self._ptr = (self._ptr + 1) % self._size

    def clear_stmemory(self):
        self.stmemory.clear()

    def sample(self, batch_size, vectorizing_fn=lambda x: x):
        bounds = np.linspace(0., 1., batch_size + 1)
        indexes = [self._sum_tree.sample(lb=bounds[i], ub=bounds[i + 1]) for i in range(batch_size)]
        priorities = np.array([self._sum_tree.get(idx) for idx in indexes]) + self._eps
        samples = [tuple(self._ltmemory[idx].values()) for idx in indexes]
        if self._beta != 0:
            priorities_pow = np.power(priorities, self._alpha)
            probs = priorities_pow / priorities_pow.sum()
            is_weights = np.power(1 / (len(self._sum_tree) * probs), self._beta)
            is_weights = is_weights / np.max(is_weights)
        else:
            is_weights = np.ones_like(indexes)
        self._beta = min(self._beta + self._beta_inc, self._beta_max)
        return vectorizing_fn(samples), indexes, is_weights

    def update_samples(self, errors, indexes):
        assert len(errors.shape) == 1 and errors.shape[0] == len(indexes)
        for error_idx, mem_idx in enumerate(indexes):
            self._sum_tree.set(mem_idx, errors[error_idx].numpy())
