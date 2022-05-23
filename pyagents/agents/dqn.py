import os
from math import log
from typing import Optional, List, Dict
import gin
import h5py
import json
import numpy as np
import tensorflow as tf
import wandb
from keras.losses import MeanSquaredError, Huber

from pyagents.agents.agent import update_target
from pyagents.agents.off_policy_agent import OffPolicyAgent
from pyagents.memory import Buffer, UniformBuffer, load_memories
from pyagents.networks import DiscreteQNetwork
from pyagents.policies import QPolicy, EpsGreedyPolicy, Policy
from copy import deepcopy


@gin.configurable
class DQNAgent(OffPolicyAgent):
    """An agent implementing the DQN algorithm. Only supports Discrete action spaces."""

    def __init__(self,
                 state_shape: tuple,
                 action_shape: tuple,
                 q_network: DiscreteQNetwork,
                 optimizer: tf.keras.optimizers.Optimizer = None,
                 gamma: float = 0.99,
                 epsilon: float = 0.1,
                 epsilon_decay: float = 0.99,
                 epsilon_min: float = 0.01,
                 target_update_period: int = 500,
                 tau: float = 1.0,
                 ddqn: bool = True,
                 buffer: Optional = 'uniform',
                 loss_fn: str = 'mse',
                 gradient_clip_norm: Optional[float] = 0.5,
                 normalize_obs: bool = True,
                 reward_scaling: float = 1.0,
                 log_dict: dict = None,
                 name: str = 'DQNAgent',
                 training: bool = True,
                 save_dir: str = './output',
                 wandb_params: Optional[dict] = None,
                 dtype: str = 'float32'):
        """Creates a DQN agent.

        Args:
            state_shape: Tuple representing the shape of the input space (observations).
            action_shape: Tuple representing the shape of the action space.
            q_network: network trained by the agent.
            optimizer: tf optimizer used for training.
            gamma: (Optional) discount factor. Defaults to 0.99.
            epsilon: (Optional) starting epsilon value for eps-greedy policy. Not used when noisy=True.
              Defaults to 0.1.
            epsilon_min: (Optional) minimum epsilon value for eps-greedy policy. Not used when noisy=True.
              Defaults to 0.01.
            epsilon_decay: (Optional) decay factor of epsilon value for eps-greedy policy. Not used when noisy=True.
              Defaults to 0.99.
            target_update_period: (Optional) number of training iterations after which target network is updated
              from online network. Defaults to 500.
            tau: (Optional) tau for polyak-averaging of target network update. Defaults to 1.0.
            ddqn: (Optional) if True, uses Double DQN loss. Defaults to True.
            buffer: (Optional) buffer to store memories. Defaults to a uniform buffer.
            normalize_obs: (Optional) if True, keeps track of running statistics to normalize observations.
              Defaults to True.
            reward_scaling: (Optional) scale factor applied to the reward. Defaults to 1.0.
            loss_fn: (Optional) loss function, either 'mse' or 'huber'. Defaults to 'mse'.
            gradient_clip_norm: (Optional) If provided, gradients are scaled so that
              the norm does not exceed this value. Defaults to 0.5.
            log_dict: (Optional) additional logger parameters.
            """
        super(DQNAgent, self).__init__(state_shape,
                                       action_shape,
                                       normalize_obs=normalize_obs,
                                       reward_scaling=reward_scaling,
                                       training=training,
                                       buffer=buffer,
                                       save_dir=save_dir,
                                       name=name,
                                       dtype=dtype)
        if optimizer is None and training:
            raise ValueError('agent cannot be trained without optimizer')
        # assert isinstance(action_shape, int), 'current implementation only supports 1D discrete action spaces'

        self._gamma = gamma * self._memory.n_step_return  # takes into account eventual n_step returns
        self._online_q_network = q_network
        self._target_q_network = deepcopy(self._online_q_network)
        self._target_update_period = target_update_period
        self._tau = tau
        self._optimizer = optimizer
        self._td_errors_loss_fn = MeanSquaredError(reduction='none') if loss_fn == 'mse' else Huber(reduction='none')
        self._train_step = 0
        self._ddqn = ddqn
        self._gradient_clip_norm = gradient_clip_norm
        self._name = name

        self.config.update({
            'gamma': self._gamma,
            'target_update_period': self._target_update_period,
            'tau': self._tau,
            'ddqn': self._ddqn,
            'gradient_clip_norm': self._gradient_clip_norm
        })

        policy = QPolicy(self._state_shape, self._action_shape, self._online_q_network)
        if self._online_q_network.noisy_layers:
            self._policy = policy
        else:
            self._policy: EpsGreedyPolicy = EpsGreedyPolicy(policy, epsilon, epsilon_decay, epsilon_min)

        if wandb_params:
            self._init_logger(wandb_params,
                              {**self.config,
                               **{f'q_net/{k}': v for k, v in self._online_q_network.get_config().items()},
                               **log_dict})

    def _wandb_define_metrics(self):
        super()._wandb_define_metrics()
        wandb.define_metric('loss', step_metric="train_step", summary="min")
        wandb.define_metric('state_values', step_metric="train_step", summary="max")
        wandb.define_metric('td_targets', step_metric="train_step", summary="max")
        wandb.define_metric('avg_state_values', step_metric="train_step", summary="max")
        wandb.define_metric('avg_td_target', step_metric="train_step", summary="max")
        wandb.define_metric('epsilon_greedy', step_metric="train_step", summary="min")

    def _loss(self, memories, weights=None):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = memories
        # boolean mask to choose only performed actions, assumes 1d action space
        mask = tf.one_hot(action_batch, self.action_shape, on_value=True, off_value=False)

        q_out = self._online_q_network(state_batch, training=True)
        preds_q_values = q_out.critic_values[mask]
        next_target_q_values = self._target_q_network(next_state_batch).critic_values

        if self._ddqn:
            # double q-learning: select actions using online network, and evaluate them using target network
            next_online_q_values = self._online_q_network(next_state_batch).critic_values
            actions = tf.one_hot(tf.math.argmax(next_online_q_values, axis=1),
                                 self.action_shape, on_value=True, off_value=False)
            bootstrap_values = next_target_q_values[actions]
        else:
            # standard dqn target
            bootstrap_values = tf.math.reduce_max(next_target_q_values, axis=1)
        bootstrap_rews = tf.stop_gradient(reward_batch + self._gamma * bootstrap_values)

        target_q_values = tf.where(done_batch, reward_batch, bootstrap_rews)
        # loss function reduces last axis, so we reshape to add a dummy one
        td_residuals = self._td_errors_loss_fn(tf.expand_dims(preds_q_values, -1), tf.expand_dims(target_q_values, -1))
        if weights is not None:  # weigh each sample by its weight before mean reduction
            td_residuals = weights * td_residuals
        loss = tf.reduce_mean(td_residuals)
        return {'loss': loss,
                'td_residuals': td_residuals,
                'q_values': preds_q_values,
                'q_targets': target_q_values}

    def _train(self, batch_size=128, *args, **kwargs):
        assert self._training, 'called train function while in evaluation mode, call toggle_training() before'
        # assert len(self._memory) > batch_size, f'batch size bigger than amount of memories'
        memories, indexes, is_weights = self._memory.sample(batch_size, vectorizing_fn=self._minibatch_to_tf)
        # resets noisy layers noise parameters (if used)
        if self._online_q_network.noisy_layers:
            self._online_q_network.reset_noise()
            self._target_q_network.reset_noise()

        # compute loss
        with tf.GradientTape() as tape:
            loss_info = self._loss(memories, weights=is_weights)
            loss = loss_info['loss']
        # use computed loss to update memories priorities (when using a prioritized buffer)
        self._memory.update_samples(tf.math.abs(loss_info['td_residuals']), indexes)
        # backward pass
        variables_to_train = self._online_q_network.trainable_weights
        grads = tape.gradient(loss, variables_to_train)
        if self._gradient_clip_norm is not None:
            grads, norm = tf.clip_by_global_norm(grads, self._gradient_clip_norm)
        grads_and_vars = list(zip(grads, variables_to_train))
        self._optimizer.apply_gradients(grads_and_vars)
        self._train_step += 1

        # periodically update target network
        if tf.math.mod(self._train_step, self._target_update_period) == 0:
            update_target(source_vars=self._online_q_network.variables,
                          target_vars=self._target_q_network.variables,
                          tau=self._tau)

        # decay epsilon of eps-greedy policy
        if isinstance(self._policy, EpsGreedyPolicy):
            self._policy.update_eps()
            epsilon = self._policy.epsilon
        else:
            epsilon = 0

        # # resets noisy layers noise parameters (if used)
        # if self._online_q_network.noisy_layers:
        #     self._online_q_network.reset_noise()
        #     self._target_q_network.reset_noise()

        # logging
        if self.is_logging:
            if self._log_gradients:
                grads_log = {f'{".".join(var.name.split("/")[1:])}': grad.numpy()
                             for grad, var in grads_and_vars}
                if self._gradient_clip_norm is not None:
                    grads_log['critic/norm'] = norm
                self._log(do_log_step=False, prefix='gradients', **grads_log)
            values_log = loss_info['q_values'].numpy()
            targets_log = loss_info['q_targets'].numpy()
            self._log(do_log_step=False, prefix='debug',
                      state_values=values_log,
                      td_targets=targets_log,
                      avg_state_value=np.mean(values_log),
                      avg_td_target=np.mean(targets_log),
                      epsilon_greedy=epsilon)
            self._log(do_log_step=True, loss=float(loss), train_step=self._train_step)

        return {'loss': float(loss)}

    def _minibatch_to_tf(self, minibatch):
        """ Given a list of experience tuples (s_t, a_t, r_t, s_t+1, done_t)
            returns list of 5 tensors batches """
        states = np.array([sample[0].reshape(self.state_shape) for sample in minibatch])
        next_states = np.array([sample[3].reshape(self.state_shape) for sample in minibatch])
        if 'obs' in self.normalizers:
            states = self.normalize('obs', states)
            next_states = self.normalize('obs', next_states)
        states_batch = tf.convert_to_tensor(states, dtype=self.dtype)
        next_states_batch = tf.convert_to_tensor(next_states, dtype=self.dtype)
        action_batch = tf.convert_to_tensor([sample[1] for sample in minibatch], dtype=tf.int32)
        reward_batch = tf.convert_to_tensor([sample[2] for sample in minibatch], dtype=self.dtype)
        done_batch = np.array([sample[4] for sample in minibatch])
        return [states_batch, action_batch, reward_batch, next_states_batch, done_batch]

    def _networks_config_and_weights(self):
        net_config = self._online_q_network.get_config()
        net_weights = self._online_q_network.get_weights()
        return [('q_net', net_config, net_weights)]

    @staticmethod
    def networks_name() -> List[tuple]:
        return [('q_net', DiscreteQNetwork)]

    @staticmethod
    def generate_input_config(
            agent_config: dict,
            networks: dict,
            load_mem: bool,
            path: str) -> Dict:
        if load_mem:
            buffer = load_memories(path)
        else:
            buffer = None
        q_net = networks['q_net']
        agent_config.update({'q_network': q_net,
                             'buffer': buffer})
        return agent_config
