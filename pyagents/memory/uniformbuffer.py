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
        self._n_step_return = n_step_return
        self._stmemory = deque(maxlen=n_step_return)
        self._ltmemory = deque(maxlen=size)
        self._config = {'size': size, 'n_step_return': n_step_return, 'type': 'uniform'}

    def __len__(self):
        return len(self._ltmemory)

    @property
    def n_step_return(self):
        return self._n_step_return

    def get_config(self):
        return self._config

    def commit_stmemory(self, fragment: np.ndarray, gamma: float = 0.99):
        states, actions, rewards, next_states, dones = fragment
        batch_size = states.shape[0]  # first dim is input batch size, i.e. n_envs
        if self._n_step_return == 1:  # no actual multi step return
            for b in range(batch_size):
                lt_experience = {'state': states[b],
                                 'action': actions[b],
                                 'reward': rewards[b],
                                 'next_state': next_states[b],
                                 'done': dones[b]}
                self.commit_ltmemory(lt_experience)
        else:
            st_experience = {'states': states, 'actions': actions, 'rewards': rewards,
                             'next_states': next_states, 'dones': dones}
            if len(self._stmemory) == self._n_step_return:  # time to compute truncated multi step return
                for b in range(batch_size):
                    s_t = self._stmemory[0]['states'][b]
                    a_t = self._stmemory[0]['actions'][b]
                    r_tpn = 0
                    for k, e_k in enumerate(self._stmemory):
                        r_tpn += (gamma ** k) * e_k['rewards'][b]
                        if e_k['dones'][b]:
                            break
                    s_tpn = e_k['next_states'][b]
                    done_tpn = e_k['dones'][b]
                    lt_experience = {'state': s_t, 'action': a_t, 'reward': r_tpn, 'next_state': s_tpn, 'done': done_tpn}
                    self.commit_ltmemory(lt_experience)
            self._stmemory.append(st_experience)

    def commit_ltmemory(self, experience):
        self._ltmemory.append(experience)

    def sample(self, batch_size, vectorizing_fn=lambda x: x):
        # no need to return samples indexes, and is_weights contains all ones (as it's not used)
        # compute samples, convert into tuple for retrocompatibily TODO switch to better dict interface
        samples = random.sample(self._ltmemory, batch_size)
        samples = list(map(lambda d: tuple(d.values()), samples))
        return vectorizing_fn(samples), [], np.ones(batch_size)
