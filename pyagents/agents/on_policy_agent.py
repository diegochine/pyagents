import abc
from abc import ABC
from typing import Optional
from collections import namedtuple

import numpy as np
import tensorflow as tf

from pyagents.agents import Agent

Trajectory = namedtuple('Trajectory', ['states', 'actions', 'rewards', 'next_states', 'dones'])


class OnPolicyAgent(Agent, ABC):
    def __init__(self,
                 state_shape: tuple,
                 action_shape: tuple,
                 training: bool,
                 save_dir: str = './output',
                 save_memories: bool = False,
                 name='OffPolicyAgent'):
        super(OnPolicyAgent, self).__init__(state_shape,
                                            action_shape,
                                            training,
                                            save_dir=save_dir,
                                            save_memories=save_memories,
                                            name=name)
        self._current_trajectory = Trajectory(actions=[], states=[], rewards=[], next_states=[], dones=[])

    @property
    def on_policy(self):
        return True

    def clear_memory(self):
        self._current_trajectory = Trajectory(actions=[], states=[], rewards=[], next_states=[], dones=[])

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
        self._current_trajectory.states.append(state)
        self._current_trajectory.actions.append(action)
        self._current_trajectory.rewards.append(reward)
        self._current_trajectory.next_states.append(next_state)
        self._current_trajectory.dones.append(done)

    def compute_returns(self):
        """Computes empirical returns of current trajectory"""
        rewards = self._current_trajectory.rewards
        dones = self._current_trajectory.dones
        returns = np.empty_like(rewards)
        returns[-1] = rewards[-1]
        for i, r in enumerate(rewards[-2::-1]):
            t = len(rewards) - i - 2
            returns[t] = r + (self.gamma * dones[t] * returns[t + 1])
        return tf.convert_to_tensor(returns.reshape(-1, 1), dtype=tf.float32)

    def compute_gae(self, state_values, next_state_values) -> tf.Tensor:
        """Computes Generalized Advantage Estimation of current trajectory.

        Args:
            state_values: critic predictions of state values for each element in the trajectory.
            next_state_values: critic predictions of next state values for each element in the trajectory.
        """
        rewards = self._current_trajectory.rewards
        dones = 1 - tf.convert_to_tensor(self._current_trajectory.dones, dtype=tf.float32)
        returns = np.zeros(len(rewards) + 1)
        for t in reversed(range(len(rewards))):
            v_t = state_values[t]
            v_tp1 = next_state_values[t]
            delta = rewards[t] + (self.gamma * dones[t] + v_tp1) - v_t
            returns[t] = delta + (self.gamma * self._lam_gae * returns[t + 1] * dones[t])
        returns = tf.convert_to_tensor(returns[:-1], dtype=tf.float32)
        return tf.reshape(returns, (-1, 1))


