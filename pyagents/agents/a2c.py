import os
from typing import Optional

import gin
import h5py
import json
import numpy as np
import tensorflow as tf
from pyagents.agents import Agent
from pyagents.networks import PolicyNetwork, ValueNetwork
from pyagents.policies import Policy
from pyagents.utils import types


@gin.configurable
class A2C(Agent):

    def __init__(self,
                 state_shape: tuple,
                 action_shape: tuple,
                 actor: PolicyNetwork,
                 critic: ValueNetwork,
                 opt: tf.keras.optimizers.Optimizer,
                 policy: Policy = None,
                 gamma: types.Float = 0.5,
                 n_steps: int = 2,
                 entropy_coef: types.Float = 0.01,
                 training=True,
                 log_dict: dict = None,
                 name='A2C',
                 wandb_params: Optional[dict] = None):
        super(A2C, self).__init__(state_shape, action_shape, training=training, name=name)

        self._actor = actor
        self._critic = critic
        self._opt = opt
        self._critic_loss_fn = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

        if policy is None:
            self._policy = self._actor.policy
        else:
            self._policy = policy

        self._n_steps = n_steps

        self.gamma = gamma
        self._entropy_coef = entropy_coef

        self._trajectory = {'states': [], 'actions': [], 'rewards': []}
        self._steps_last_update = 0

        self.config.update({
            'gamma': self.gamma,
            'n_steps': self._n_steps,
            'entropy_coef': self._entropy_coef,
            **{f'actor/{k}': v for k, v in self._actor.get_config().items()},
            **{f'critic/{k}': v for k, v in self._critic.get_config().items()}
        })
        if log_dict is not None:
            self.config.update(**log_dict)

        if wandb_params:
            self._init_logger(wandb_params)

    def clear_memory(self):
        self._trajectory = {'states': [], 'actions': [], 'rewards': []}
        self._steps_last_update = 0

    def remember(self, state, action, reward):
        self._trajectory['states'].append(state)
        self._trajectory['actions'].append(action)
        self._trajectory['rewards'].append(reward)
        self._steps_last_update += 1

    def _loss(self, memories):
        states, actions, delta = memories
        critic_values = self._critic(states)
        output = self._actor(inputs=states)
        log_prob = self.policy.log_prob(output, actions)
        entropy_loss = tf.reduce_mean(self.policy.entropy(output))
        policy_loss = -tf.reduce_sum((log_prob * (delta - critic_values)))
        critic_loss = self._critic_loss_fn(delta, critic_values)
        return policy_loss, critic_loss, entropy_loss

    def _compute_n_step_returns(self, next_state):
        rewards = self._trajectory['rewards']
        returns = np.empty_like(rewards)
        critic_estimate = (self.gamma ** self._n_steps) * self._critic(next_state.reshape(1, -1))
        disc_r = 0
        for i, r_k in enumerate(rewards[::-1]):
            k = len(returns) - i - 1
            disc_r += (self.gamma ** k) * r_k
            returns[len(returns) - i - 1] = disc_r + critic_estimate
        return tf.convert_to_tensor(returns, dtype=tf.float32)

    def _train(self, next_state, batch_size=None):
        if self._steps_last_update == self._n_steps:
            states = tf.convert_to_tensor(self._trajectory['states'], dtype=tf.float32)
            actions = tf.convert_to_tensor(self._trajectory['actions'], dtype=tf.float32)

            delta = tf.stop_gradient(self._compute_n_step_returns(next_state))

            # train
            with tf.GradientTape() as tape:
                policy_loss, critic_loss, entropy_loss = self._loss((states, actions, tf.stop_gradient(delta)))
                loss = policy_loss + critic_loss - self._entropy_coef * entropy_loss

            assert not tf.math.is_inf(loss) and not tf.math.is_nan(loss)
            grads = tape.gradient(loss, self.trainable_variables)
            self._opt.apply_gradients(list(zip(grads, self.trainable_variables)))
            self.clear_memory()
            return {'policy_loss': float(policy_loss), 'critic_loss': float(critic_loss)}

    def save(self, path):
        pass

    @classmethod
    def load(cls, path, ver):
        pass