import abc
from abc import ABC
from typing import Optional

import numpy as np
import tensorflow as tf

from pyagents.agents import Agent
from pyagents.memory import Buffer, UniformBuffer, PrioritizedBuffer


class OffPolicyAgent(Agent, ABC):
    def __init__(self,
                 state_shape: tuple,
                 action_shape: tuple,
                 training: bool,
                 gamma: float = 0.99,
                 buffer: Optional[Buffer] = None,
                 normalize_obs: bool = True,
                 reward_scaling: float = 1.0,
                 save_dir: str = './output',
                 save_memories: bool = False,
                 name='OffPolicyAgent',
                 dtype: str = 'float32'):
        super(OffPolicyAgent, self).__init__(state_shape,
                                             action_shape,
                                             training,
                                             normalize_obs=normalize_obs,
                                             reward_scaling=reward_scaling,
                                             save_dir=save_dir,
                                             save_memories=save_memories,
                                             name=name,
                                             dtype=dtype)
        if isinstance(buffer, Buffer):
            buffer.set_save_dir(self._save_dir)
            self._memory: Buffer = buffer
        elif buffer == 'uniform' or buffer is None:  # default
            self._memory: Buffer = UniformBuffer(save_dir=self._save_dir)
        elif buffer == 'prioritized':
            self._memory: Buffer = PrioritizedBuffer(save_dir=self._save_dir)
        else:
            raise ValueError(f'unrecognized buffer param {buffer}')
        self._config.update({f'buffer/{k}': v for k, v in self._memory.get_config().items()})

        self._gamma = gamma
        # take into account eventual n_step returns
        self._gamma_n = tf.constant(gamma ** self._memory.n_step_return, dtype=self.dtype)
        self._config.update({'gamma': gamma})


    @property
    def on_policy(self):
        return False

    @property
    def memory_len(self):
        return len(self._memory)

    def init(self, envs, env_config=None, min_memories=None, actions=None, *args, **kwargs):
        super().init(envs, env_config=env_config, *args, **kwargs)
        if min_memories is None:
            min_memories = self._memory.get_config()['size']
        s_t = envs.reset()
        for _ in range(min_memories // self.num_envs):
            if actions is not None:
                a_t = np.random.choice(actions, 1)
            else:
                a_t = envs.action_space.sample()
            s_tp1, r_t, done, info = envs.step(a_t)
            for i, single_step in enumerate(info):
                # handle TimeLimit wrapper
                if 'TimeLimit.truncated' in single_step:
                    done[i] = not info[i]['TimeLimit.truncated']
            self.remember(s_t, a_t, r_t, s_tp1, done)
            s_t = s_tp1

    def remember(self, state: np.ndarray, action, reward: float, next_state: np.ndarray, done: bool, *args, **kwargs) -> None:
        """Saves piece of memory."""
        reward = reward * self.reward_scaling
        self._memory.commit_stmemory((state, action, reward, next_state, done), gamma=self._gamma)

    @abc.abstractmethod
    def _minibatch_to_tf(self, minibatch):
        """ Given a list of experience tuples (s_t, a_t, r_t, s_t+1, done_t)
            returns list of 5 tensors batches """
        pass

    def _do_save_memories(self):
        self._memory.save()
