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
from pyagents.utils import json_utils, types
from pyagents.agents import Agent
from pyagents.memory import Buffer, UniformBuffer, load_memories
from pyagents.networks import DiscreteQNetwork
from pyagents.policies import QPolicy, EpsGreedyPolicy, Policy
from copy import deepcopy


@gin.configurable
class DQNAgent(OffPolicyAgent):

    def __init__(self,
                 state_shape: tuple,
                 action_shape: tuple,
                 q_network: DiscreteQNetwork,
                 optimizer: tf.keras.optimizers.Optimizer = None,
                 policy: Policy = None,
                 gamma: types.Float = 0.99,
                 epsilon: types.Float = 0.1,
                 epsilon_decay: types.Float = 0.99,
                 epsilon_min: types.Float = 0.01,
                 target_update_period: int = 500,
                 tau: types.Float = 1.0,
                 ddqn: bool = True,
                 buffer: Optional[Buffer] = None,
                 loss_fn: str = 'mse',
                 gradient_clip_norm: Optional[types.Float] = 0.5,
                 log_dict: dict = None,
                 name: str = 'DQNAgent',
                 training: bool = True,
                 save_dir: str = './output',
                 wandb_params: Optional[dict] = None):
        super(DQNAgent, self).__init__(state_shape,
                                       action_shape,
                                       training=training,
                                       buffer=buffer,
                                       save_dir=save_dir,
                                       name=name)
        if optimizer is None and training:
            raise ValueError('agent cannot be trained without optimizer')

        self._gamma = gamma
        self._online_q_network = q_network
        self._target_q_network = deepcopy(self._online_q_network)
        self._target_update_period = target_update_period
        self._tau = tau
        self._optimizer = optimizer
        self._td_errors_loss_fn = MeanSquaredError(reduction='none') if loss_fn == 'mse' else Huber(reduction='none')
        self._train_step = tf.Variable(0, trainable=False, name="train step counter")
        self._ddqn = ddqn
        self._gradient_clip_norm = gradient_clip_norm
        self._name = name

        self.config.update({
            'gamma': self._gamma,
            'target_update_period': self._target_update_period,
            'tau': self._tau,
            'ddqn': self._ddqn,
        })

        if policy is None:
            policy = QPolicy(self._state_shape, self._action_shape, self._online_q_network)
            self._policy: EpsGreedyPolicy = EpsGreedyPolicy(policy, epsilon, epsilon_decay, epsilon_min)
        else:
            self._policy = policy

        if wandb_params:
            self._init_logger(wandb_params,
                              {**self.config,
                               **{f'q_net/{k}': v for k, v in self._online_q_network.get_config().items()},
                               **log_dict})

    def _wandb_define_metrics(self):
        wandb.define_metric('loss', step_metric="train_step", summary="min")

    def _loss(self, memories):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = memories
        current_q_values = self._online_q_network(state_batch)
        next_target_q_values = self._target_q_network(next_state_batch)

        if self._ddqn:
            next_online_q_values = self._online_q_network(next_state_batch)
            action_idx = tf.convert_to_tensor([[b, a]
                                               for b, a in enumerate(tf.math.argmax(next_online_q_values, axis=1))])
            tmp_rewards = tf.stop_gradient(reward_batch + self._gamma * tf.gather_nd(next_target_q_values, action_idx))
        else:
            tmp_rewards = tf.stop_gradient(
                reward_batch + self._gamma * tf.math.reduce_max(next_target_q_values, axis=1))

        target_values = tf.where(done_batch, reward_batch, tmp_rewards)
        target_values = tf.stack([target_values for _ in range(self._action_shape)], axis=1)
        update_idx = tf.convert_to_tensor([[i == a for i in range(self._action_shape)] for a in action_batch])
        target_q_values = tf.where(update_idx, target_values, current_q_values)
        # reshape qvals because of reduction axis
        td_loss = self._td_errors_loss_fn(tf.expand_dims(current_q_values, -1), tf.expand_dims(target_q_values, -1))
        td_loss = tf.reduce_mean(td_loss, axis=-1)
        return td_loss

    def _train(self, batch_size=128, *args, **kwargs):
        self._memory.commit_ltmemory()
        assert self._training, 'called train function while in evaluation mode, call toggle_training() before'
        assert len(self._memory) > batch_size, f'batch size bigger than amount of memories'
        memories, indexes, is_weights = self._memory.sample(batch_size, vectorizing_fn=self._minibatch_to_tf)
        with tf.GradientTape() as tape:
            td_loss = self._loss(memories)
            loss = tf.reduce_mean(is_weights * td_loss)
        # use computed loss to update memories priorities (when using a prioritized buffer)
        self._memory.update_samples(tf.math.abs(td_loss), indexes)
        variables_to_train = self._online_q_network.trainable_weights
        grads = tape.gradient(loss, variables_to_train)
        if self._gradient_clip_norm is not None:
            grads, norm = tf.clip_by_global_norm(grads, self._gradient_clip_norm)
        grads_and_vars = list(zip(grads, variables_to_train))
        self._optimizer.apply_gradients(grads_and_vars)
        self._train_step.assign_add(1)
        if tf.math.mod(self._train_step, self._target_update_period) == 0:
            update_target(source_vars=self._online_q_network.variables,
                          target_vars=self._target_q_network.variables,
                          tau=self._tau)
        # the following only for epsgreedy policies
        # TODO make it more generic
        self._policy.update_eps()
        return {'td_loss': float(loss)}

    def _minibatch_to_tf(self, minibatch):
        """ Given a list of experience tuples (s_t, a_t, r_t, s_t+1, done_t)
            returns list of 5 tensors batches """
        state_batch = tf.convert_to_tensor([sample[0].reshape(self.state_shape)
                                            for sample in minibatch])
        action_batch = tf.convert_to_tensor([sample[1] for sample in minibatch])
        reward_batch = tf.convert_to_tensor([sample[2] for sample in minibatch])
        new_state_batch = tf.convert_to_tensor([sample[3].reshape(self.state_shape)
                                                for sample in minibatch])
        done_batch = np.array([sample[4] for sample in minibatch])
        return [state_batch, action_batch, reward_batch, new_state_batch, done_batch]

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
