import os
from typing import Optional
import gin
import h5py
import json
import numpy as np
import tensorflow as tf
from keras.losses import Huber, mean_squared_error
from pyagents.agents import Agent
from pyagents.networks import ActorNetwork, ValueNetwork
from pyagents.policies import Policy
from pyagents.utils import types


@gin.configurable
class VPG(Agent):

    def __init__(self,
                 state_shape: tuple,
                 action_shape: tuple,
                 actor: ActorNetwork,
                 critic: ValueNetwork,
                 actor_opt: tf.keras.optimizers.Optimizer,
                 critic_opt: tf.keras.optimizers.Optimizer,
                 policy: Policy = None,
                 gamma: types.Float = 0.5,
                 standardize: bool = True,
                 entropy_coef: types.Float = 0.1,
                 training=True,
                 name='A2C'):
        super(A2C, self).__init__(state_shape, action_shape, training=training, name=name)

        self._actor = actor
        self._critic = critic
        self._opt = actor_opt
        self._critic_opt = critic_opt
        self._critic_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM) #tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

        if policy is None:
            self._policy = self._actor.policy
        else:
            self._policy = policy

        self.gamma = gamma
        self._entropy_coef = entropy_coef

        self._trajectory = {'states': [], 'actions': [], 'rewards': []}
        self._standardize = standardize

    def clear_memory(self):
        self._trajectory = {'states': [], 'actions': [], 'rewards': []}

    def remember(self, state, action, reward):
        self._trajectory['states'].append(state)
        self._trajectory['actions'].append(action)
        self._trajectory['rewards'].append(reward)

    def _loss(self, memories):
        pass

    def _train(self, batch_size=None):
        states = self._trajectory['states']
        actions = self._trajectory['actions']
        rewards = self._trajectory['rewards']

        returns = np.empty_like(rewards)
        returns[-1] = rewards[-1]
        for i, r in enumerate(rewards[-2::-1]):
            returns[len(rewards) - i - 2] = r + self.gamma * returns[len(rewards) - i - 1]

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns.reshape(-1, 1), dtype=tf.float32)

        if self._standardize:
            returns = ((returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + 1e-5))

        # train
        with tf.GradientTape() as tape:
            critic_values = self._critic(states)
            adv = returns - critic_values
            output = self._actor(inputs=states)
            log_prob = self.policy.log_prob(output, actions)
            entropy_loss = self.policy.entropy(output)
            policy_loss = -tf.reduce_sum((log_prob * adv) + self._entropy_coef * entropy_loss)
            critic_loss = self._critic_loss(returns, critic_values)
            # critic_loss = 0.5 * tf.reduce_mean(tf.square(returns - critic_values) + 0.1 * entropy_loss, axis=0)
            loss = policy_loss + critic_loss

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