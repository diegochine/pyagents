from typing import Optional
import gin
import h5py
import numpy as np
import tensorflow as tf
from keras.losses import Huber, mean_squared_error
from tensorflow.python.keras.saving.saved_model import json_utils

from agents.agent import Agent
from memory import Buffer, UniformBuffer
from networks import QNetwork
from policies import QPolicy, EpsGreedyPolicy
from utils import types
from copy import deepcopy


@gin.configurable
class DQNAgent(Agent):

    CFG_Q_NET = 'qnet.json'
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
                 save_dir: str = './output'):
        super(DQNAgent, self).__init__(state_shape, action_shape, name=name)
        self._save_dir = f'{save_dir}/{name}'
        if buffer is not None:
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

        policy = QPolicy(self._state_shape, self._action_shape, self._online_q_network)
        self._policy = EpsGreedyPolicy(policy, epsilon, epsilon_decay, epsilon_min)

    @property
    def memory_len(self):
        return len(self._memory.stmemory)

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
            tmp_rewards = tf.stop_gradient(reward_batch + self._gamma * tf.math.reduce_max(next_target_q_values, axis=1))

        target_values = tf.where(done_batch, reward_batch, tmp_rewards)
        target_values = tf.stack([target_values for _ in range(self._action_shape)], axis=1)
        update_idx = tf.convert_to_tensor([[i == a for i in range(self._action_shape)] for a in action_batch])
        target_q_values = tf.where(update_idx, target_values, current_q_values)
        td_loss = self._td_errors_loss_fn(current_q_values, target_q_values)
        return td_loss

    def _train(self, batch_size):
        self._memory.commit_ltmemory()
        memories, indexes = self._memory.sample(batch_size, vectorizing_fn=self._minibatch_to_tf)
        with tf.GradientTape() as tape:
            td_loss = self._loss(memories)
            loss = tf.reduce_mean(td_loss)
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

    def memory_init(self, env, max_steps, min_memories):
        while self.memory_len <= min_memories:
            s = env.reset()
            done = False
            step = 0
            while not done and step < max_steps:
                a = env.action_space.sample()
                new_state, r, done, _ = env.step(a)
                self.remember(s, a, r, new_state, done)
                s = new_state
                step += 1

    def save(self, v=1):
        fname = f'{self._name}_v{v}'
        self._memory.save(fname)
        net_config = self._online_q_network.get_config()
        net_weights = self._online_q_network.get_weights()
        json.dump(net_config, open(f'{self._save_dir}/{self.CFG_Q_NET}', 'w'))

        f = h5py.File(f'{self._save_dir}/{self.W_Q_NET}', mode='w')
        try:
            for k, v in net_config.items():
                if isinstance(v, (dict, list, tuple)):
                    f.attrs[k] = json.dumps(
                        v, default=json_utils.get_json_type).encode('utf8')
                else:
                    f.attrs[k] = v

            model_weights_group = f.create_group('model_weights')
            model_layers = model.layers
            save_weights_to_hdf5_group(model_weights_group, model_layers)

            # TODO(b/128683857): Add integration tests between tf.keras and external
            # Keras, to avoid breaking TF.js users.
            if (include_optimizer and model.optimizer and
                    not isinstance(model.optimizer, optimizer_v1.TFOptimizer)):
                save_optimizer_weights_to_hdf5_group(f, model.optimizer)

            f.flush()
        finally:
            f.close()

    @classmethod
    def load(cls, path, optimizer=None):
        # TODO https://github.com/keras-team/keras/blob/be4cef42ab21d85398fb6930ec5419a3de8a7d71/keras/saving/hdf5_format.py
        q_net_cfg = json.load(open(f'{path}/{cls.CFG_Q_NET}', 'r'))
