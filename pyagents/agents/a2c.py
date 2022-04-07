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
    """ An agent implementing a synchronous, deterministic version of the A3C algorithm """

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
                 entropy_coef: types.Float = 1e-3,
                 gradient_clip_norm: Optional[types.Float] = 0.5,
                 training: bool = True,
                 log_dict: dict = None,
                 name: str = 'A2C',
                 wandb_params: Optional[dict] = None):
        """Creates an A2C agent.

        Args:
            state_shape: Tuple representing the shape of the input space (observations).
            action_shape: Tuple representing the shape of the action space.
            actor_critic: A network comprising both a policy and a value function.
            opt: Optimizer to use for training.
            policy: (Optional) The policy the agent should follow. Defaults to policy network's policy.
            gamma: (Optional) Discount factor. Defaults to 0.5.
            standardize: (Optional) If True, standardizes delta before computing gradient update. Defaults to True.
            n_step_return: (Optional) Number of steps before computing multi-step returns. Defaults to 1.
            lam_gae: (Optional) Lambda for Generalized Advantage Estimation. # TODO not implemented
            acc_grads_steps: (Optional) Simulates A3C asynchronous parallel learning
              by accumulating gradients over multiple backward passes. Defaults to 0 (no accumulation).
            entropy_coef: (Optional) Coefficient applied to entropy loss. Defaults to 1e-3.
            gradient_clip_norm: (Optional) Global norm for gradient clipping, pass None to disable. Defaults to 0.5.
            training: (Optional) If True, agent is in training phase. Defaults to True.
            log_dict: (Optional) Additional dict of config parameters that should be logged by WandB. Defaults to None.
            name: (Optional) Name of the agent.
            wandb_params: (Optional) Dict of parameters to enable WandB logging. Defaults to None.
        """
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

        self._trajectory = {'states': [], 'actions': [], 'rewards': [], 'next_states': [], 'dones': []}
        self._steps_last_update = 0
        self._acc_grads_steps = acc_grads_steps
        self._acc_grads = None

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
        """Defines WandB metrics.

        For A2C, they are: policy network loss; critic network loss; policy distribution entropy loss.
        """
        wandb.define_metric('policy_loss', step_metric="train_step", summary="min")
        wandb.define_metric('critic_loss', step_metric="train_step", summary="min")
        wandb.define_metric('entropy_loss', step_metric="train_step", summary="min")

    def clear_memory(self):
        self._trajectory = {'states': [], 'actions': [], 'rewards': [], 'next_states': [], 'dones': []}

    def remember(self, state: np.ndarray, action, reward: float, next_state: np.ndarray, done: bool) -> None:
        self._trajectory['states'].append(state)
        self._trajectory['actions'].append(action)
        self._trajectory['rewards'].append(reward)
        self._trajectory['next_states'].append(next_state)
        self._trajectory['dones'].append(done)

    def _loss(self, memories):
        states, actions, delta = memories
        actor_out, critic_values = self._actor_critic(inputs=states)
        log_prob = self._policy.log_prob(actor_out, actions)
        entropy_loss = tf.reduce_sum(self._policy.entropy(actor_out))
        policy_loss = -tf.reduce_sum((log_prob * (delta - tf.stop_gradient(critic_values))))
        critic_loss = self._critic_loss_fn(delta, critic_values)
        return policy_loss, critic_loss, entropy_loss

    def _compute_n_step_returns(self, next_state: np.ndarray, done: bool) -> tf.Tensor:
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

    def _train(self, batch_size=None, *args, **kwargs) -> dict:
        done, next_state = self._trajectory['dones'][-1], self._trajectory['next_states'][-1]
        if done or len(self._trajectory['rewards']) == self._n_step_return:  # perform training step
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
            if self._acc_grads is None:
                self._acc_grads = np.array(grads)
            else:
                self._acc_grads += np.array(grads)

            # simulate A3C async update of gradients by multiple parallel learners
            if self._steps_last_update == self._acc_grads_steps:
                self._opt.apply_gradients(list(zip(self._acc_grads, self.trainable_variables)))
                self._steps_last_update = 0
                self._acc_grads = None
            else:
                self._steps_last_update += 1

            self.clear_memory()
            return {'policy_loss': float(policy_loss),
                    'critic_loss': float(critic_loss),
                    'entropy_loss': float(entropy_loss)}

    def save(self, path):
        pass

    @classmethod
    def load(cls, path, ver):
        pass
