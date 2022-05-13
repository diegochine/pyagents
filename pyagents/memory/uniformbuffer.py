import random
import gin
from collections import deque

import numpy as np

from pyagents.memory.buffer import Buffer


@gin.configurable
class UniformBuffer(Buffer):

    def __init__(self,
                 save_dir=None,
                 size=50000,
                 n_step_return=1):
        super().__init__(save_dir)
        self._ltmemory = deque(maxlen=size)
        self._n = n_step_return - 1
        self._config = {'size': size}

    def __len__(self):
        return len(self.ltmemory)

    def get_config(self):
        return self._config

    def commit_stmemory(self, fragment):
        states, actions, rewards, next_states, dones = fragment
        batch_size = states.shape[0]  # first dim is input batch size, i.e. n_envs
        for b in range(batch_size):
            lt_experience = (states[b], actions[b], rewards[b], next_states[b], dones[b])
            self._ltmemory.append(lt_experience)

    def commit_ltmemory(self, **kwargs):
        for mem in self.stmemory:
            self.ltmemory.append(mem)
        self.clear_stmemory()

    def clear_stmemory(self):
        self.stmemory.clear()

    def sample(self, batch_size, vectorizing_fn=lambda x: x):
        # no need to return samples indexes, and is_weights contains all ones (as it's not used)
        return vectorizing_fn(random.sample(list(self._ltmemory), batch_size)), [], np.ones(batch_size)
