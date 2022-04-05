import os
from typing import Optional
import gin
import h5py
import json
import numpy as np
import tensorflow as tf
import wandb
from pyagents.agents import Agent
from pyagents.networks import PolicyNetwork, ValueNetwork
from pyagents.policies import Policy
from pyagents.utils import types


@gin.configurable
class VPG(Agent):

    def __init__(self,
                 state_shape: tuple,
                 action_shape: tuple,
                 actor: PolicyNetwork,
                 actor_opt: tf.keras.optimizers.Optimizer,
                 critic: ValueNetwork = None,
                 critic_opt: tf.keras.optimizers.Optimizer = None,
                 policy: Policy = None,
                 gamma: types.Float = 0.5,
                 standardize: bool = True,
                 critic_value_coef: types.Float = 0.5,
                 entropy_coef: types.Float = 1e-4,
                 gradient_clip_norm: Optional[float] = None,
                 training: bool = True,
                 save_dir: str = './output',
                 log_dict: dict = None,
                 name: str = 'A2C',
                 wandb_params: Optional[dict] = None):
        super(VPG, self).__init__(state_shape, action_shape, training=training, name=name)

        self._actor = actor
        self._critic = critic
        self._actor_opt = actor_opt
        if self._critic is not None:
            assert critic_opt is not None, 'must provide a critic optimizer when providing a critic network'
        self._critic_opt = critic_opt
        self._critic_loss_fn = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

        if policy is None:
            self._policy = self._actor.policy
        else:
            self._policy = policy

        self.gamma = gamma
        self._critic_value_coef = critic_value_coef
        self._entropy_coef = entropy_coef

        self._trajectory = {'states': [], 'actions': [], 'rewards': []}
        self._standardize = standardize
        self._gradient_clip_norm = gradient_clip_norm

        self.config.update({
            'gamma': self.gamma,
            'standardization': self._standardize,
            'critic_value_coef': self._critic_value_coef,
            'entropy_coef': self._entropy_coef,
            **{f'actor/{k}': v for k, v in self._actor.get_config().items()}
        })
        if self._critic is not None:
            self.config.update(**{f'critic/{k}': v for k, v in self._critic.get_config().items()})
        else:
            self.config['critic'] = False
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

    def remember(self, state, action, reward):
        self._trajectory['states'].append(state)
        self._trajectory['actions'].append(action)
        self._trajectory['rewards'].append(reward)

    def _loss(self, memories):
        states, actions, returns = memories
        actor_output = self._actor(inputs=states)
        log_prob = self.policy.log_prob(actor_output, actions)
        log_prob = tf.reshape(log_prob, (-1, 1))
        if self._critic is not None:
            critic_values = self._critic(states)
            adv = (returns - tf.stop_gradient(critic_values))
            if self._standardize:
                adv = ((adv - tf.math.reduce_mean(adv)) / (tf.math.reduce_std(adv) + self.eps))
            policy_loss = -tf.reduce_sum((log_prob * adv))
            critic_loss = self._critic_value_coef * self._critic_loss_fn(returns, critic_values)
        else:
            policy_loss = -tf.reduce_sum((log_prob * returns))
            critic_loss = 0
        entropy_loss = self._entropy_coef * tf.reduce_mean(self.policy.entropy(actor_output))
        return policy_loss, critic_loss, entropy_loss

    def _train(self, batch_size=None):
        # convert inputs to tf tensors and compute returns
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
            returns = ((returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + self.eps))

        # training: forward and loss computation
        with tf.GradientTape(persistent=True) as tape:
            policy_loss, critic_loss, entropy_loss = self._loss((states, actions, tf.stop_gradient(returns)))
            loss = policy_loss + critic_loss - entropy_loss
        assert not tf.math.is_inf(loss) and not tf.math.is_nan(loss)

        # training: backward pass
        actor_grads = tape.gradient(loss, self._actor.trainable_variables)
        critic_grads = tape.gradient(loss, self._critic.trainable_variables)
        if self._gradient_clip_norm is not None:
            actor_grads, actor_norm = tf.clip_by_global_norm(actor_grads, self._gradient_clip_norm)
            critic_grads, critic_norm = tf.clip_by_global_norm(critic_grads, self._gradient_clip_norm)
        actor_grads_and_vars = list(zip(actor_grads, self._actor.trainable_variables))
        critic_grads_and_vars = list(zip(critic_grads, self._critic.trainable_variables))
        self._actor_opt.apply_gradients(actor_grads_and_vars)
        self._critic_opt.apply_gradients(critic_grads_and_vars)
        del tape
        self.clear_memory()
        self._train_step += 1

        # logging
        if self.is_logging:
            actor_grads_log = {f'actor/{".".join(var.name.split("/")[1:])}': grad.numpy()
                               for grad, var in actor_grads_and_vars}
            critic_grads_log = {f'critic/{".".join(var.name.split("/")[1:])}': grad.numpy()
                                for grad, var in critic_grads_and_vars}
            if self._gradient_clip_norm is not None:
                actor_grads_log['actor/norm'] = actor_norm
                critic_grads_log['critic/norm'] = critic_norm
            self._log(do_log_step=False, prefix='gradients', **actor_grads_log, **critic_grads_log)

        return {'policy_loss': float(policy_loss),
                'critic_loss': float(critic_loss),
                'entropy_loss': float(entropy_loss),
                'train_step': self._train_step}

    def save(self, path):
        pass

    @classmethod
    def load(cls, path, ver):
        pass
