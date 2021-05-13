import gin
import numpy as np
import tensorflow as tf
from keras import Input
from keras.models import Model
from keras.layers import Dense
from keras.optimizers import Adam
from keras.regularizers import l1_l2, l2
from keras.losses import Huber

from agents.agent import Agent
from memory import Memory
from networks import network
import config as cfg
from copy import deepcopy

@gin.configurable
class DQNAgent(Agent):

    def __init__(self,
                 state_shape: tuple,
                 action_shape: tuple,
                 q_network: network,
                 optimizer: tf.keras.optimizers.Optimizer,
                 gamma: float = 0.7,
                 epsilon: float = 0.01,
                 memory_size: int = 10000):
        super(DQNAgent, self).__init__(state_shape, action_shape)
        self._memory = Memory(size_long=memory_size)
        self._gamma = gamma
        self._epsilon = epsilon
        self._q_network = q_network
        self._target_q_network = deepcopy(self._q_network)
        self._optimizer = optimizer
        self._td_errors_loss_fn = Huber(reduction=tf.keras.losses.Reduction.NONE)

    def act(self, state):
        if np.random.rand() <= self._epsilon:
            return np.random.randint(self._action_shape)
        qvals = self._q_network(state.reshape(1, *state.shape))
        return np.argmax(qvals)

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
        self._memory.commit_stmemory({'state': state, 'action': action, 'reward': reward,
                                     'next_state': next_state, 'done': done})

    def _loss(self, memories):
        losses = []
        for experience in memories:
            q_values = self._q_network(experience['state'])
            if not experience['done']:
                next_q_values = self._target_q_network(experience['next_state'])
                td_targets = tf.stop_gradient(experience['reward'] + self._gamma * next_q_values)
            else:
                td_targets = experience['reward']
            td_loss = self._td_errors_loss_fn(td_targets, q_values)
            losses.append(td_loss)
        return tf.reduce_mean(losses)

    def _train(self, batch_size):
        self._memory.commit_ltmemory()
        if len(self._memory) > 128:
            memories = self._memory.sample(batch_size)
            with tf.GradientTape() as tape:
                loss = self._loss(memories)
            variables_to_train = self._q_network.trainable_weights
            grads = tape.gradient(loss, variables_to_train)
            grads_and_vars = list(zip(grads, variables_to_train))
            self._optimizer.apply_gradients(grads_and_vars)
            return loss

    # TODO update target network
    # TODO save and load agent

class DQNAgentv1(Agent):

    def __init__(self, state_shape, action_shape, model=None, gamma=cfg.GAMMA, learning_rate=cfg.LEARNING_RATE,
                 epsilon=cfg.EPSILON, epsilon_decay=cfg.EPSILON_DECAY, epsilon_min=cfg.EPSILON_MIN,
                 memory_size=cfg.MEMORY_SIZE, min_memories=cfg.MIN_MEMORIES):

        super(DQNAgent, self).__init__(state_shape, action_shape)

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.min_memories = min_memories

        if model is None:
            self.model = self._build_model()
        else:
            self.model = model
            self.model.compile(loss='mse', optimizer=Adam(self.learning_rate))

    def _build_model(self):

        inputs = Input(shape=self.state_shape)
        x = Dense(16, activation='relu',
                  kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                  bias_regularizer=l2(1e-4))(inputs)
        x = Dense(32, activation='relu',
                  kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                  bias_regularizer=l2(1e-4))(x)
        x = Dense(16, activation='relu',
                  kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                  bias_regularizer=l2(1e-4))(x)
        # output layer, linear because the output are qvals
        outputs = Dense(self.action_shape, activation='linear')(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='mse', optimizer=Adam(self.learning_rate))

        return model



    def act(self, state):
        """
        :param state: current state
        :return: the best action according to current policy
        """
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_shape)
        qvals = self.model(state.reshape(1, *state.shape))
        return np.argmax(qvals)

    def replay(self, batch_size):
        """
        Performs experience replay
        :param batch_size:
        :return:
        """
        q_value = -100
        for fragment in reversed(self.memory.last_episode()):
            if fragment['done']:
                q_value = fragment['reward']
                fragment['q_value'] = q_value
            else:
                q_value = fragment['reward'] + self.gamma * q_value
                fragment['q_value'] = q_value

        self.memory.commit_ltmemory()

        if len(self.memory) > self.min_memories:
            minibatch = self.memory.sample(batch_size)

            for fragment in minibatch:
                target = fragment['q_value']
                action = fragment['action']
                state = fragment['state']
                # adding batch size dimension to state
                # TODO: perform this operation in batch
                state = state.reshape(1, *state.shape)
                target_f = self.model.predict(state).reshape(1, self.action_shape)
                target_f[0][action] = target

                self.model.fit(state, target_f, epochs=1, verbose=0)

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save(name)
