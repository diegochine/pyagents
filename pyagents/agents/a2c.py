import os
from typing import Optional
import gin
import h5py
import json
import numpy as np
import tensorflow as tf
import wandb
from pyagents.agents import Agent
from pyagents.networks import ActorCriticNetwork
from pyagents.policies import Policy
from pyagents.utils import types


@gin.configurable
class A2C(Agent):

    def __init__(self,
                 state_shape: tuple,
                 action_shape: tuple,
                 actor_critic: ActorCriticNetwork,
                 opt: tf.keras.optimizers.Optimizer,
                 policy: Policy = None,
                 gamma: types.Float = 0.5,
                 standardize: bool = True,
                 n_step_return: int = 1,
                 lam_gae: float = 1,
                 acc_grads_steps: int = 0,
                 entropy_coef: types.Float = 0.01,
                 gradient_clip_norm: types.Float = 0.5,
                 training=True,
                 log_dict: dict = None,
                 name='A2C',
                 wandb_params: Optional[dict] = None):
        super(A2C, self).__init__(state_shape, action_shape, training=training, name=name)

        self._actor_critic = actor_critic
        self._opt = opt
        self._critic_loss_fn = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

        if policy is None:
            self._policy = self._actor_critic.get_policy()
        else:
            self._policy = policy

        self._n_step_return = n_step_return

        self.gamma = gamma
        self._entropy_coef = entropy_coef
        self._lam_gae = lam_gae
        self._standardize = standardize
        self._gradient_clip_norm = gradient_clip_norm

        self._trajectory = {'states': [], 'actions': [], 'rewards': []}
        self._steps_last_update = 0
        self._acc_grads_steps = acc_grads_steps

        self.config.update({
            'gamma': self.gamma,
            'n_step_return': self._n_step_return,
            'entropy_coef': self._entropy_coef,
            **{f'actor_critic/{k}': v for k, v in self._actor_critic.get_config().items()}
        })
        if log_dict is not None:
            self.config.update(**log_dict)

        if wandb_params:
            self._init_logger(wandb_params)

    def _wandb_define_metrics(self):
        wandb.define_metric('policy_loss', step_metric="train_step", summary="min")
        wandb.define_metric('critic_loss', step_metric="train_step", summary="min")
        wandb.define_metric('entropy_loss', step_metric="train_step", summary="min")

    def clear_memory(self):
        self._trajectory = {'states': [], 'actions': [], 'rewards': []}
        self._steps_last_update = 0

    def remember(self, state, action, reward):
        self._trajectory['states'].append(state)
        self._trajectory['actions'].append(action)
        self._trajectory['rewards'].append(reward)

    def _loss(self, memories):
        states, actions, delta = memories
        actor_out, critic_values = self._actor_critic(inputs=states)
        log_prob = self._policy.log_prob(actor_out, actions)
        entropy_loss = tf.reduce_mean(self._policy.entropy(actor_out))
        policy_loss = -tf.reduce_sum((log_prob * (delta - tf.stop_gradient(critic_values))))
        critic_loss = self._critic_loss_fn(delta, critic_values)
        return policy_loss, critic_loss, entropy_loss

    def _compute_n_step_returns(self, next_state, done):
        rewards = self._trajectory['rewards']
        returns = np.empty_like(rewards)
        if done:
            disc_r = 0
        else:
            next_state = tf.convert_to_tensor(next_state.reshape(1, *next_state.shape))
            _, disc_r = self._actor_critic(next_state)
            disc_r = float(disc_r)
        for i, r_k in enumerate(rewards[::-1]):
            k = len(returns) - i - 1
            disc_r += (self.gamma ** k) * r_k
            returns[k] = disc_r
        return tf.convert_to_tensor(returns, dtype=tf.float32)

    def _train(self, batch_size=None, *args, **kwargs):
        if len(self._trajectory['rewards']) == self._n_step_return:  # perform training step
            assert 'next_state' in kwargs and 'done' in kwargs, 'must pass next state and done flag to train A2C agent'
            next_state, done = kwargs['next_state'], kwargs['done']
            states = tf.convert_to_tensor(self._trajectory['states'], dtype=tf.float32)
            actions = tf.convert_to_tensor(self._trajectory['actions'], dtype=tf.float32)

            delta = tf.stop_gradient(self._compute_n_step_returns(next_state, done))
            if self._standardize:
                delta = ((delta - tf.math.reduce_mean(delta)) / (tf.math.reduce_std(delta) + self.eps))
            # train
            with tf.GradientTape() as tape:
                policy_loss, critic_loss, entropy_loss = self._loss((states, actions, tf.stop_gradient(delta)))
                loss = policy_loss + critic_loss - self._entropy_coef * entropy_loss

            assert not tf.math.is_inf(loss) and not tf.math.is_nan(loss)
            grads = tape.gradient(loss, self.trainable_variables)
            if self._gradient_clip_norm is not None:
                grads, norm = tf.clip_by_global_norm(grads, self._gradient_clip_norm)

            self._opt.apply_gradients(list(zip(grads, self.trainable_variables)))
            self._steps_last_update += 1
            self.clear_memory()
            return {'policy_loss': float(policy_loss), 'critic_loss': float(critic_loss)}

    def save(self, path):
        pass

    @classmethod
    def load(cls, path, ver):
        pass
