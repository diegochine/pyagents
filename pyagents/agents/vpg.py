import os
from typing import Optional, List, Dict
import gin
import h5py
import json
import numpy as np
import tensorflow as tf
import wandb
from pyagents.agents.on_policy_agent import OnPolicyAgent
from pyagents.networks import PolicyNetwork, ValueNetwork
from pyagents.policies import Policy
from pyagents.utils import types


@gin.configurable
class VPG(OnPolicyAgent):
    """An agent implementing the REINFORCE algorithm, also known as Vanilla Policy Gradient."""

    def __init__(self,
                 state_shape: tuple,
                 action_shape: tuple,
                 actor: PolicyNetwork,
                 actor_opt: tf.keras.optimizers.Optimizer = None,
                 critic: ValueNetwork = None,
                 critic_opt: tf.keras.optimizers.Optimizer = None,
                 policy: Policy = None,
                 gamma: types.Float = 0.99,
                 standardize: bool = True,
                 lam_gae: float = 0.9,
                 critic_value_coef: types.Float = 0.5,
                 entropy_coef: types.Float = 0,
                 gradient_clip_norm: Optional[float] = None,
                 training: bool = True,
                 rew_gamma: float = 0.0,
                 save_dir: str = './output',
                 log_dict: dict = None,
                 name: str = 'VPG',
                 wandb_params: Optional[dict] = None):
        """Creates a VPG agent.

                Args:
                    state_shape: Tuple representing the shape of the input space (observations).
                    action_shape: Tuple representing the shape of the action space.
                    actor: The policy network.
                    actor_opt: Optimizer to use for training the actor.
                    critic: (Optional) The critic network to use as baseline for computing advantages.
                      If None, no baseline is used. Defaults to None.
                    critic_opt: (Optional) Optimizer to use for training the critic. Must be provided when
                      a critic is also provided. Defaults to None.
                    critic_value_coef: (Optional) Coefficient applied on the value function during loss computation.
                    policy: (Optional) The policy the agent should follow. Defaults to policy network's policy.
                    gamma: (Optional) Discount factor. Defaults to 0.5.
                    standardize: (Optional) If True, standardizes delta before computing gradient update.
                      Defaults to False.
                    lam_gae: (Optional) Lambda for Generalized Advantage Estimation. Defaults to 0.9.
                    entropy_coef: (Optional) Coefficient applied to entropy loss. Defaults to 1e-3.
                    gradient_clip_norm: (Optional) Global norm for gradient clipping, pass None to disable.
                      Defaults to 0.5.
                    training: (Optional) If True, agent is in training phase. Defaults to True.
                    log_dict: (Optional) Additional dict of config parameters that should be logged by WandB.
                      Defaults to None.
                    name: (Optional) Name of the agent.
                    wandb_params: (Optional) Dict of parameters to enable WandB logging. Defaults to None.
                """
        super(VPG, self).__init__(state_shape, action_shape, rew_gamma=rew_gamma, lam_gae=lam_gae,
                                  training=training, save_dir=save_dir, name=name)
        if actor_opt is None and training:
            raise ValueError('agent cannot be trained without optimizer')

        self._actor = actor
        self._baseline = critic
        self._actor_opt = actor_opt
        if self._baseline is not None:
            assert critic_opt is not None or not training, 'must provide a critic optimizer when providing a critic network'
        self._critic_opt = critic_opt
        self._critic_loss_fn = tf.keras.losses.MeanSquaredError()

        if policy is None:
            self._policy = self._actor.get_policy()
        else:
            self._policy = policy

        self.gamma = gamma
        self._critic_value_coef = critic_value_coef
        self._entropy_coef = entropy_coef
        self._standardize = standardize
        self._gradient_clip_norm = gradient_clip_norm

        self.config.update({
            'gamma': self.gamma,
            'standardize': self._standardize,
            'critic_value_coef': self._critic_value_coef,
            'entropy_coef': self._entropy_coef
        })

        if wandb_params:
            if log_dict is None:
                log_dict = {}
            self._init_logger(wandb_params,
                              {**self.config,
                               **{f'actor/{k}': v for k, v in self._actor.get_config().items()},
                               **{f'baseline/{k}': v for k, v in self._baseline.get_config().items()},
                               **log_dict})
        self._actor(tf.ones((1, *state_shape))), self._baseline(tf.ones((1, *state_shape)))  # TODO remove

    def _wandb_define_metrics(self):
        wandb.define_metric('policy_loss', step_metric="train_step", summary="min")
        wandb.define_metric('critic_loss', step_metric="train_step", summary="min")
        wandb.define_metric('entropy_loss', step_metric="train_step", summary="min")

    def _loss(self, memories):
        states, actions, returns, delta = memories
        dist_params = self._actor(inputs=states).dist_params
        log_prob = self.get_policy().log_prob(dist_params, actions)
        log_prob = tf.reshape(log_prob, (-1, 1))
        if self._baseline is not None:
            critic_values = self._baseline(states).critic_values
            policy_loss = -tf.reduce_mean((log_prob * delta))
            critic_loss = self._critic_value_coef * self._critic_loss_fn(returns, critic_values)
        else:
            policy_loss = -tf.reduce_mean((log_prob * returns))
            critic_loss = 0
        entropy_loss = self._entropy_coef * tf.reduce_mean(self.get_policy().entropy(dist_params))
        return policy_loss, critic_loss, entropy_loss

    def _train(self, batch_size, update_rounds, *args, **kwargs):
        # convert inputs to tf tensors and compute delta
        states = tf.convert_to_tensor(self._memory['states'], dtype=tf.float32)
        actions = tf.convert_to_tensor(self._memory['actions'], dtype=tf.float32)
        states = tf.reshape(states, (self._rollout_size, -1))
        actions = tf.reshape(actions, (self._rollout_size, -1))

        returns = self.compute_returns()
        state_values = self._baseline(states).critic_values

        if self._lam_gae > 0:
            next_states = tf.convert_to_tensor(self._memory['next_states'], dtype=tf.float32)
            next_states = tf.reshape(next_states, (self._rollout_size, -1))
            next_state_values = self._baseline(next_states).critic_values
            delta = self.compute_gae(state_values=state_values, next_state_values=next_state_values)
        else:
            delta = returns - tf.stop_gradient(state_values)

        if self._standardize:
            returns = ((returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + self.eps))
            delta = ((delta - tf.math.reduce_mean(delta)) / (tf.math.reduce_std(delta) + self.eps))

        delta = tf.reshape(delta, (self._rollout_size, 1))
        returns = tf.reshape(returns, (self._rollout_size, 1))

        indexes = np.arange(states.shape[0])
        for _ in range(update_rounds):
            np.random.shuffle(indexes)
            # training: iterate minibatches
            for start in range(0, states.shape[0], batch_size):
                end = start + batch_size
                batch_indexes = indexes[start:end]
                s_b = tf.gather(states, batch_indexes, axis=0)
                a_b = tf.gather(actions, batch_indexes, axis=0)
                d_b = tf.gather(delta, batch_indexes, axis=0)
                r_b = tf.gather(returns, batch_indexes, axis=0)

                # training: forward and loss computation
                with tf.GradientTape(persistent=True) as tape:
                    policy_loss, critic_loss, entropy_loss = self._loss((s_b, a_b,
                                                                         tf.stop_gradient(r_b), tf.stop_gradient(d_b)))
                    loss = policy_loss + critic_loss - entropy_loss
                assert not tf.math.is_inf(loss) and not tf.math.is_nan(loss)

                # training: backward pass
                actor_grads = tape.gradient(loss, self._actor.trainable_variables)
                critic_grads = tape.gradient(loss, self._baseline.trainable_variables)
                if self._gradient_clip_norm is not None:
                    actor_grads, actor_norm = tf.clip_by_global_norm(actor_grads, self._gradient_clip_norm)
                    critic_grads, critic_norm = tf.clip_by_global_norm(critic_grads, self._gradient_clip_norm)
                actor_grads_and_vars = list(zip(actor_grads, self._actor.trainable_variables))
                critic_grads_and_vars = list(zip(critic_grads, self._baseline.trainable_variables))
                self._actor_opt.apply_gradients(actor_grads_and_vars)
                self._critic_opt.apply_gradients(critic_grads_and_vars)
                del tape
                self._train_step += 1

                # logging
                if self.is_logging:
                    if self._log_gradients:
                        actor_grads_log = {f'actor/{".".join(var.name.split("/")[1:])}': grad.numpy()
                                           for grad, var in actor_grads_and_vars}
                        critic_grads_log = {f'critic/{".".join(var.name.split("/")[1:])}': grad.numpy()
                                            for grad, var in critic_grads_and_vars}
                        if self._gradient_clip_norm is not None:
                            actor_grads_log['actor/norm'] = actor_norm
                            critic_grads_log['critic/norm'] = critic_norm
                        self._log(do_log_step=False, prefix='gradients', **actor_grads_log, **critic_grads_log)
                    losses_log = {'policy_loss': float(policy_loss),
                                  'critic_loss': float(critic_loss),
                                  'entropy_loss': float(entropy_loss),
                                  'train_step': self._train_step}
                    self._log(do_log_step=True, **losses_log)

        self.clear_memory()

        return {'policy_loss': float(policy_loss),
                'critic_loss': float(critic_loss),
                'entropy_loss': float(entropy_loss)}

    def _networks_config_and_weights(self):
        a = [('actor_net', self._actor.get_config(), self._actor.get_weights())]
        if self._baseline:
            a.append(('critic_net', self._baseline.get_config(), self._baseline.get_weights()))
        return a

    @staticmethod
    def networks_name() -> List[tuple]:
        return [('actor_net', PolicyNetwork), ('critic_net', ValueNetwork)]

    @staticmethod
    def generate_input_config(
            agent_config: dict,
            networks: dict,
            load_mem: bool,
            path: str) -> Dict:
        actor = networks['actor_net']
        if 'critic_net' in networks:
            critic = networks['critic_net']
        else:
            critic = None
        agent_config.update({'actor': actor,
                             'critic': critic})
        return agent_config
