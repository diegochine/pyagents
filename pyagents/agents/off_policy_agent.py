import abc
from abc import ABC
from typing import Optional

import numpy as np

from pyagents.agents import Agent
from pyagents.memory import Buffer, UniformBuffer


class OffPolicyAgent(Agent, ABC):
    def __init__(self,
                 state_shape: tuple,
                 action_shape: tuple,
                 training: bool,
                 buffer: Optional[Buffer] = None,
                 save_dir: str = './output',
                 save_memories: bool = False,
                 name='OffPolicyAgent'):
        super(OffPolicyAgent, self).__init__(state_shape,
                                             action_shape,
                                             training,
                                             save_dir=save_dir,
                                             save_memories=save_memories,
                                             name=name)
        if buffer is not None:
            buffer.set_save_dir(self._save_dir)
            self._memory: Buffer = buffer
        else:
            self._memory: Buffer = UniformBuffer(save_dir=self._save_dir)

    @property
    def memory_len(self):
        return len(self._memory)

    def memory_init(self, env, max_steps, min_memories, actions=None):
        print('Collecting initial memories')
        while self.memory_len < min_memories:
            s = env.reset()
            done = False
            step = 0
            self._memory.commit_ltmemory()
            while not done and step < max_steps:
                if actions:
                    a = np.random.choice(actions, 1)
                else:
                    a = env.action_space.sample()
                new_state, r, done, _ = env.step(a)
                self.remember(s, a, r, new_state, done)
                s = new_state
                step += 1

    def remember(self, state: np.ndarray, action, reward: float, next_state: np.ndarray, done: bool) -> None:
        """
        Saves piece of memory
        :param state: state at current timestep
        :param action: action at current timestep
        :param reward: reward at current timestep
        :param next_state: state at next timestep
        :param done: whether the episode has ended
        :return:
        """
        self._memory.commit_stmemory([state, action, reward, next_state, done])

    @abc.abstractmethod
    def _minibatch_to_tf(self, minibatch):
        """ Given a list of experience tuples (s_t, a_t, r_t, s_t+1, done_t)
            returns list of 5 tensors batches """
        pass

    def _do_save_memories(self):
        self._memory.save()
