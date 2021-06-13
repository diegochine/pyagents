import os
from typing import Optional
import gin
import h5py
import json
import numpy as np
import tensorflow as tf
from keras.losses import Huber, mean_squared_error
from utils import json_utils
from agents.agent import Agent
from memory import Buffer, UniformBuffer, load_memories
from networks import QNetwork
from policies import QPolicy, EpsGreedyPolicy
from utils import types
from copy import deepcopy


@gin.configurable
class DQNAgent(Agent):
    W_Q_NET = 'qnet.hdf5'

    def __init__(self,
                 state_shape: tuple,
                 action_shape: tuple,
                 q_network: QNetwork,
                 optimizer: tf.keras.optimizers.Optimizer,
                 gamma: types.Float = 0.5,
                 epsilon: types.Float = 0.1,
                 epsilon_decay: types.Float = 0.98,
                 epsilon_min: types.Float = 0.01,
                 target_update_period: int = 500,
                 tau: types.Float = 1.0,
                 ddqn: bool = True,
                 buffer: Optional[Buffer] = None,
                 name: str = 'DQNAgent',
                 training: bool = True,
                 save_dir: str = './output'):
        super(DQNAgent, self).__init__(state_shape, action_shape, name=name)
        self._save_dir = os.path.join(save_dir, name)
        if not os.path.isdir(self._save_dir):
            os.makedirs(self._save_dir)
        if buffer is not None:
            buffer.set_save_dir(self._save_dir)
            self._memory: Buffer = buffer
        else:
            self._memory: Buffer = UniformBuffer(save_dir=self._save_dir)
        self._gamma = gamma
        self._online_q_network = q_network
        self._target_q_network = deepcopy(self._online_q_network)
        self._target_update_period = target_update_period
        self._tau = tau
        self._optimizer = optimizer
        self._td_errors_loss_fn = mean_squared_error  # Huber(reduction=tf.keras.losses.Reduction.NONE)
        self._train_step = tf.Variable(0, trainable=False, name="train step counter")
        self._ddqn = ddqn
        self._training = training
        self._name = name

        self._config = {
            'gamma': self._gamma,
            'target_update_period': self._target_update_period,
            'tau': self._tau,
            'ddqn': self._ddqn,
            'name': self._name
        }

        policy = QPolicy(self._state_shape, self._action_shape, self._online_q_network)
        self._policy = EpsGreedyPolicy(policy, epsilon, epsilon_decay, epsilon_min)

    @property
    def memory_len(self):
        return len(self._memory.ltmemory)

    @property
    def state_shape(self):
        return self._state_shape

    @property
    def epsilon(self):
        return self._policy.epsilon

    def act(self, state):
        return self._policy.act(state)

    def remember(self, state, action, reward, next_state, done):
        """
        Saves piece of memory
        :param state: state at current timestep
        :param action: action at current timestep
        :param reward: reward at current timestep
        :param next_state: state at next timestep
        :param done: whether the episode has ended
        :return:
        """
        self._memory.commit_stmemory([state, action, reward, next_state, done])

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
        td_loss = self._td_errors_loss_fn(current_q_values, target_q_values)
        return td_loss

    def _train(self, batch_size):
        self._memory.commit_ltmemory()
        if self._training:
            memories, indexes, is_weights = self._memory.sample(batch_size, vectorizing_fn=self._minibatch_to_tf)
            with tf.GradientTape() as tape:
                td_loss = self._loss(memories)
                loss = tf.reduce_mean(is_weights * td_loss)
            # use computed loss to update memories priorities (when using a prioritized buffer)
            self._memory.update_samples(tf.math.abs(td_loss), indexes)
            variables_to_train = self._online_q_network.trainable_weights
            grads = tape.gradient(loss, variables_to_train)
            grads_and_vars = list(zip(grads, variables_to_train))
            self._optimizer.apply_gradients(grads_and_vars)
            self._train_step.assign_add(1)
            if tf.math.mod(self._train_step, self._target_update_period) == 0:
                self._update_target()
            # the following only for epsgreedy policies
            # TODO make it more generic
            self._policy.update_eps()
            return loss

    def _update_target(self):
        source_variables = self._online_q_network.variables
        target_variables = self._target_q_network.variables
        for (sv, tv) in zip(source_variables, target_variables):
            tv.assign((1 - self._tau) * tv + self._tau * sv)

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

    def memory_init(self, env, max_steps, min_memories, actions=None):
        while self.memory_len <= min_memories:
            s = env.reset()
            done = False
            step = 0
            self._memory.commit_ltmemory()
            while not done and step < max_steps:
                if actions:
                    a = np.random.choice(actions, 1)
                else:
                    a = env.action_space.sample()
                new_state, r, done, _ = env.step(a)
                self.remember(s, a, r, new_state, done)
                s = new_state
                step += 1

    def save(self, ver, include_optimizer=False):
        self._memory.save()
        net_config = self._online_q_network.get_config()
        net_weights = self._online_q_network.get_weights()

        f = h5py.File(f'{self._save_dir}/{self.W_Q_NET}_v{ver}', mode='w')
        try:
            agent_group = f.create_group('agent')
            for k, v in self._config.items():
                agent_group.attrs[k] = v

            net_config_group = f.create_group('net_config')
            for k, v in net_config.items():
                if isinstance(v, (dict, list, tuple)):
                    net_config_group.attrs[k] = json.dumps(v, default=json_utils.get_json_type).encode('utf8')
                else:
                    net_config_group.attrs[k] = v

            net_weights_group = f.create_group('net_weights')
            for i, lay_weights in enumerate(net_weights):
                net_weights_group.create_dataset(f'net_weights{i}', data=lay_weights)

            if include_optimizer and self._optimizer:
                pass
                # TODO save_optimizer_weights_to_hdf5_group(f, model.optimizer)

            f.flush()
        finally:
            f.close()

    @classmethod
    def load(cls, path, preprocessing_layers=None, optimizer=None, **kwargs):
        if optimizer is None:
            optimizer = tf.keras.optimizers.RMSprop(momentum=0.1)
        f = h5py.File(f'{path}/{cls.W_Q_NET}', mode='r')
        agent_group = f['agent']
        agent_config = {}
        for k, v in agent_group.attrs.items():
            agent_config[k] = v

        net_config = {}
        net_config_group = f['net_config']
        for k, v in net_config_group.attrs.items():
            if isinstance(v, bytes):
                net_config[k] = json_utils.decode(v)
            else:
                net_config[k] = v
        net_weights_group = f['net_weights']
        net_weights = [weights[()] for name, weights in net_weights_group.items()]
        f.close()
        q_net = QNetwork.from_config(net_config)
        q_net(tf.ones((1, net_config['state_shape'])))
        q_net.set_weights(net_weights)
        buffer = load_memories(path)
        agent_config.update({'state_shape': net_config['state_shape'],
                             'action_shape': net_config['action_shape'],
                             'q_network': q_net,
                             'optimizer': optimizer,
                             'buffer': buffer,
                             **kwargs})
        return cls(**agent_config)
