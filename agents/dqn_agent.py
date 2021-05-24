import gin
import numpy as np
import tensorflow as tf
from keras.losses import Huber, mean_squared_error

from agents.agent import Agent
from memory import Memory
from networks import Network
from policies import QPolicy, EpsGreedyPolicy
from copy import deepcopy


@gin.configurable
class DQNAgent(Agent):

    def __init__(self,
                 state_shape: tuple,
                 action_shape: tuple,
                 q_network: Network,
                 optimizer: tf.keras.optimizers.Optimizer,
                 gamma: float = 0.5,
                 epsilon: float = 0.1,
                 epsilon_decay: float = 0.98,
                 epsilon_min: float = 0.01,
                 target_update_period: int = 500,
                 memory_size: int = 10000):
        super(DQNAgent, self).__init__(state_shape, action_shape)
        self._memory = Memory(size_long=memory_size)
        self._gamma = gamma
        self._q_network = q_network
        self._target_q_network = deepcopy(self._q_network)
        self._target_update_period = target_update_period
        self._optimizer = optimizer
        self._td_errors_loss_fn = mean_squared_error  # Huber(reduction=tf.keras.losses.Reduction.NONE)
        self._train_step = tf.Variable(0, trainable=False, name="train step counter")

        policy = QPolicy(self._state_shape, self._action_shape, self._q_network)
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
        state_batch, action_batch, reward_batch, new_state_batch, done_batch = memories

        next_q_values = self._target_q_network(new_state_batch)
        q_values = self._q_network(state_batch)

        target_values = tf.where(done_batch, reward_batch,
                                 tf.stop_gradient(reward_batch + self._gamma * tf.math.reduce_max(next_q_values, axis=1)))
        target_values = tf.stack([target_values for _ in range(self._action_shape)], axis=1)
        update_idx = tf.convert_to_tensor([[i == a for i in range(self._action_shape)] for a in action_batch])
        target_q_values = tf.where(update_idx, target_values, q_values)
        td_loss = self._td_errors_loss_fn(q_values, target_q_values)
        return tf.reduce_mean(td_loss)

    def _train(self, batch_size):
        self._memory.commit_ltmemory()
        memories = self._memory.sample(batch_size, vectorizing_fn=self._vectorize_samples)
        with tf.GradientTape() as tape:
            loss = self._loss(memories)
        variables_to_train = self._q_network.trainable_weights
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
        source_variables = self._q_network.variables
        target_variables = self._target_q_network.variables
        for (sv, tv) in zip(source_variables, target_variables):
            tv.assign(sv)

    def _vectorize_samples(self, mini_batch):
        state_batch = tf.convert_to_tensor([sample[0].reshape(self.state_shape)
                                            for sample in mini_batch])
        action_batch = tf.convert_to_tensor([sample[1] for sample in mini_batch])
        reward_batch = tf.convert_to_tensor([sample[2] for sample in mini_batch])
        new_state_batch = tf.convert_to_tensor([sample[3].reshape(self.state_shape)
                                                for sample in mini_batch])
        done_batch = np.array([sample[4] for sample in mini_batch])
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

    def save(self, path, name='DQNAgent', v=1):
        fname = f'{name}_{v}'
        self._memory.save(fname)
