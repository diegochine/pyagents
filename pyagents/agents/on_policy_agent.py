from abc import ABC

import gym
import numpy as np
import tensorflow as tf

from pyagents.agents import Agent


class OnPolicyAgent(Agent, ABC):

    MEMORY_KEYS = {'states', 'actions', 'rewards', 'next_states', 'dones', 'logprobs'}

    def __init__(self,
                 state_shape: tuple,
                 action_shape: tuple,
                 training: bool,
                 lam_gae: float = 0.9,
                 save_dir: str = './output',
                 save_memories: bool = False,
                 name='OffPolicyAgent'):
        super(OnPolicyAgent, self).__init__(state_shape,
                                            action_shape,
                                            training,
                                            save_dir=save_dir,
                                            save_memories=save_memories,
                                            name=name)
        self._memory = {k: None for k in self.MEMORY_KEYS}  # keeps track of current trajectories
        self._rollout_size = None  # size of above container, #steps * #envs
        self._step = 0  # step for indexing above memory
        self._lam_gae = lam_gae


    @property
    def on_policy(self):
        return True

    def init(self, envs, rollout_steps, *args, **kwargs):
        """Initializes memory (i.e. trajectories storage)."""
        assert isinstance(envs, gym.vector.VectorEnv), 'envs must be instance of VectorEnv even for a single instance'
        if isinstance(envs.single_action_space, gym.spaces.Discrete):
            self._memory['actions'] = np.zeros((rollout_steps, envs.num_envs))
        elif isinstance(envs.single_action_space, gym.spaces.Box):
            self._memory['actions'] = np.zeros((rollout_steps, envs.num_envs) + self.action_shape)
        else:
            raise NotImplementedError(f'unsupported action space {envs.single_action_space}')

        self._memory['states'] = np.zeros((rollout_steps, envs.num_envs) + self.state_shape)
        self._memory['next_states'] = np.zeros((rollout_steps, envs.num_envs) + self.state_shape)
        self._memory['rewards'] = np.zeros((rollout_steps, envs.num_envs))
        self._memory['dones'] = np.zeros((rollout_steps, envs.num_envs))
        self._memory['logprobs'] = np.zeros((rollout_steps, envs.num_envs))
        self._rollout_size = rollout_steps * envs.num_envs

    def clear_memory(self):
        self._memory = {k: np.zeros_like(self._memory[k]) for k in self.MEMORY_KEYS}
        self._step = 0

    def remember(self,
                 state: np.ndarray,
                 action: np.array,
                 reward: float,
                 next_state: np.ndarray,
                 done: bool,
                 logprob: float = 0.0,
                 *args, **kwargs) -> None:
        """
        Saves piece of memory

        Args:
            state: state at current timestep
            action: action at current timestep
            reward: reward at current timestep
            next_state: state at next timestep
            done: whether the episode has ended
            logprob:
        """
        self._memory['states'][self._step] = state
        self._memory['actions'][self._step] = action
        self._memory['rewards'][self._step] = reward
        self._memory['next_states'][self._step] = next_state
        self._memory['dones'][self._step] = done
        self._memory['logprobs'][self._step] = logprob
        self._step += 1

    def compute_returns(self):
        """Computes empirical returns of current trajectory"""
        rewards = self._memory['rewards'].reshape(-1)
        dones = 1 - self._memory['dones'].reshape(-1)
        returns = np.empty_like(rewards)
        returns[-1] = rewards[-1]
        for i, r in enumerate(rewards[-2::-1]):
            t = len(rewards) - i - 2
            returns[t] = r + (self.gamma * dones[t] * returns[t + 1])
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)
        return tf.reshape(returns, (-1, 1))

    def compute_gae(self, state_values, next_state_values) -> tf.Tensor:
        """Computes Generalized Advantage Estimation of current trajectory.

        Args:
            state_values: critic predictions of state values for each element in the trajectory.
            next_state_values: critic predictions of next state values for each element in the trajectory.
        """
        state_values = state_values.numpy()
        next_state_values = next_state_values.numpy()
        rewards = np.reshape(self._memory['rewards'], -1)
        if 'rewards' in self.normalizers:
            rewards = self.normalize('rewards', rewards)
        dones = 1 - np.reshape(self._memory['dones'], -1)
        advantages = np.zeros(len(rewards))
        advantages[-1] = rewards[-1] + (self.gamma * dones[-1] + next_state_values[-1]) - state_values[-1]
        for t in reversed(range(len(rewards) - 1)):
            v_t = state_values[t]
            v_tp1 = next_state_values[t]
            delta = rewards[t] + (self.gamma * dones[t] + v_tp1) - v_t
            advantages[t] = delta + (self.gamma * self._lam_gae * advantages[t + 1] * dones[t])
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        return tf.stop_gradient(tf.reshape(advantages, (-1, 1)))